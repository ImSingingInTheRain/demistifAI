(function () {
  const root = document.getElementById("root");

  if (!root) {
    return;
  }

  const resolveStreamlit = () =>
    window.Streamlit || window.streamlit_component_lib?.Streamlit || null;

  const bootstrap = (Streamlit, initialArgs) => {
    let activeController = null;
    let mountedDomId = null;
    let lastInputValue = null;
    let lastSerializedSignature = null;
    let mountedMarkupKey = null;
    let mountedMarkupSignature = null;

    const canSetValue =
      Streamlit && typeof Streamlit.setComponentValue === "function";
    const canSetFrameHeight =
      Streamlit && typeof Streamlit.setFrameHeight === "function";
    const canSetReady =
      Streamlit && typeof Streamlit.setComponentReady === "function";
    const canListen = Boolean(
      Streamlit &&
        Streamlit.events &&
        typeof Streamlit.events.addEventListener === "function"
    );
    const renderEvent = Streamlit ? Streamlit.RENDER_EVENT : null;

    const defaultState = () => ({ text: "", submitted: false, ready: false });

    const coerceString = (value, fallback = "") =>
      typeof value === "string" ? value : fallback;

    const coerceNumber = (value, fallback = 0) => {
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : fallback;
    };

    const sanitizeClass = (value) => {
      if (typeof value !== "string" || !value) {
        return null;
      }
      const trimmed = value.trim();
      if (!trimmed) {
        return null;
      }
      const cleaned = trimmed.replace(/[^-a-zA-Z0-9_]/g, "");
      return cleaned || null;
    };

    const sanitizeSegments = (rawSegments) => {
      if (!Array.isArray(rawSegments)) {
        return [];
      }
      return rawSegments.map((line) => {
        if (!Array.isArray(line)) {
          return [];
        }
        return line.map((segment) => ({
          t: coerceString(segment && segment.t, ""),
          c: sanitizeClass(segment && segment.c),
        }));
      });
    };

    const fallbackSegmentsFromLines = (lines, suffix) =>
      lines.map((line) => {
        const trimmed = line.trim();
        if (line.startsWith("$ ")) {
          return [{ t: line, c: `cmd-${suffix}` }];
        }
        if (/^ERROR\b/i.test(trimmed)) {
          return [{ t: line, c: `err-${suffix}` }];
        }
        return [{ t: line, c: null }];
      });

    const toLinesWithNewline = (arr) =>
      arr.map((value) => {
        const text = coerceString(value, "");
        return text.endsWith("\n") ? text : `${text}\n`;
      });

    const escapeHtml = (text) =>
      coerceString(text, "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

    const pushState = (state) => {
      if (canSetValue) {
        Streamlit.setComponentValue({ value: state });
      }
    };

    const stableStringify = (value) => {
      if (value === null || value === undefined) {
        return "null";
      }
      if (typeof value !== "object") {
        return JSON.stringify(value);
      }
      if (Array.isArray(value)) {
        return `[${value.map((item) => stableStringify(item)).join(",")}]`;
      }
      const keys = Object.keys(value).sort();
      const entries = keys.map(
        (key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`
      );
      return `{${entries.join(",")}}`;
    };

    const notifyResize = (height) => {
      if (!canSetFrameHeight) {
        return;
      }
      const coerced = Number(height);
      if (Number.isFinite(coerced) && coerced > 0) {
        Streamlit.setFrameHeight(coerced + 24);
      } else {
        const fallbackHeight = document.body ? document.body.scrollHeight : 0;
        Streamlit.setFrameHeight(fallbackHeight);
      }
    };

    const initializeTerminal = (rootNode, options) => {
      let payload = options.payload || {};
      const suffix = coerceString(payload.suffix, "");
      if (!suffix) {
        pushState(defaultState());
        return null;
      }

      const pre = rootNode.querySelector(`.term-body-${suffix}`);
      const caret = rootNode.querySelector(`.caret-${suffix}`);
      const inputWrap =
        rootNode.querySelector(
          `.term-input-wrap-${suffix}:not([data-secondary="true"])`
        ) || rootNode.querySelector(`.term-input-wrap-${suffix}`);
      const input =
        rootNode.querySelector(
          `.term-input-${suffix}:not([data-secondary="true"])`
        ) || rootNode.querySelector(`.term-input-${suffix}`);

      let normalisedLines = [];
      const syncPayload = (nextPayload) => {
        payload = nextPayload || {};
        const raw = Array.isArray(payload.lines)
          ? payload.lines.map((line) => (line == null ? "" : String(line)))
          : [];
        normalisedLines = toLinesWithNewline(raw);
        const placeholderText = coerceString(payload.placeholder, "");
        const inputLabel = coerceString(payload.inputLabel, "");
        const inputAriaLabel = coerceString(payload.inputAriaLabel, "");
        const inputHint = coerceString(payload.inputHint, "");
        const inputId = coerceString(payload.inputId, "");
        const inputHintId = coerceString(payload.inputHintId, "");
        if (input) {
          if (input.placeholder !== placeholderText) {
            input.placeholder = placeholderText;
          }
          if (inputAriaLabel) {
            input.setAttribute("aria-label", inputAriaLabel);
          }
          if (inputHintId) {
            input.setAttribute("aria-describedby", inputHintId);
          }
          if (inputId) {
            input.id = inputId;
          }
        }
        if (inputWrap) {
          const labelEl = inputWrap.querySelector(
            `label.sr-only-${suffix}[for="${inputId}"]`
          );
          if (labelEl && labelEl.textContent !== inputLabel) {
            labelEl.textContent = inputLabel;
          }
        }
        if (inputHintId) {
          const hintEl = rootNode.querySelector(`#${inputHintId}`);
          if (hintEl && hintEl.textContent !== inputHint) {
            hintEl.textContent = inputHint;
          }
        }
      };

      syncPayload(payload);

      if (!pre) {
        pushState(defaultState());
        return null;
      }

      const controller = {
        domId: coerceString(payload.domId, ""),
        suffix,
        destroy: () => {},
        updateInputValue: () => {},
        updateAcceptKeystrokes: () => {},
        replaceSerializedSegments: () => {},
        appendSerializedSegments: () => {},
        getLineCount: () => 0,
      };

      const acceptOverride = options.acceptKeystrokes;
      let acceptKeystrokes =
        typeof acceptOverride === "boolean"
          ? acceptOverride
          : Boolean(payload.acceptKeystrokes);
      const debounceMs = Math.max(0, coerceNumber(payload.debounceMs, 150));

      const initialText = input
        ? coerceString(payload.inputValue, input.value)
        : coerceString(payload.inputValue, "");
      if (input) {
        input.value = initialText;
      }

      const state = {
        text: input ? coerceString(input.value, "") : initialText,
        submitted: false,
        ready: false,
      };
      let wasReady = false;
      let debounceHandle = null;
      let typingHandle = null;
      let rafHandle = null;
      const cleanupFns = [];

      const copyState = () => ({
        text: state.text,
        submitted: state.submitted,
        ready: state.ready,
      });

      let lastSentSnapshot = copyState();

      const statesEqual = (a, b) =>
        Boolean(a) &&
        Boolean(b) &&
        a.text === b.text &&
        a.submitted === b.submitted &&
        a.ready === b.ready;

      const flushState = () => {
        if (!canSetValue) {
          return;
        }
        const snapshot = copyState();
        if (statesEqual(snapshot, lastSentSnapshot)) {
          return;
        }
        lastSentSnapshot = snapshot;
        pushState(snapshot);
      };

      const sendState = (immediate) => {
        if (immediate) {
          flushState();
          return;
        }
        if (!canSetValue) {
          return;
        }
        if (debounceHandle !== null) {
          window.clearTimeout(debounceHandle);
        }
        debounceHandle = window.setTimeout(() => {
          flushState();
        }, debounceMs);
      };

      const syncReadyUi = () => {
        rootNode.setAttribute("data-ready", state.ready ? "true" : "false");
        if (inputWrap) {
          const active = state.ready && acceptKeystrokes;
          inputWrap.setAttribute("data-active", active ? "true" : "false");
          inputWrap.setAttribute("aria-hidden", active ? "false" : "true");
        }
        if (input) {
          if (state.ready && acceptKeystrokes) {
            input.disabled = false;
            input.setAttribute("aria-disabled", "false");
            if (!wasReady) {
              wasReady = true;
              window.requestAnimationFrame(() => {
                try {
                  input.focus({ preventScroll: true });
                } catch (_err) {
                  input.focus();
                }
              });
            }
          } else {
            input.disabled = true;
            input.setAttribute("aria-disabled", "true");
          }
        }
      };

      controller.updateAcceptKeystrokes = (value) => {
        const next = Boolean(value);
        if (next !== acceptKeystrokes) {
          acceptKeystrokes = next;
          syncReadyUi();
        }
      };

      const setState = (updates, opts) => {
        const immediate = Boolean(opts && opts.immediate);
        const prevReady = state.ready;
        if (updates && typeof updates === "object") {
          if (
            Object.prototype.hasOwnProperty.call(updates, "text") &&
            typeof updates.text === "string"
          ) {
            state.text = updates.text;
          }
          if (
            Object.prototype.hasOwnProperty.call(updates, "submitted") &&
            typeof updates.submitted === "boolean"
          ) {
            state.submitted = updates.submitted;
          }
          if (
            Object.prototype.hasOwnProperty.call(updates, "ready") &&
            typeof updates.ready === "boolean"
          ) {
            state.ready = updates.ready;
          }
        }
        if (prevReady !== state.ready) {
          syncReadyUi();
        }
        sendState(immediate);
      };

      syncReadyUi();
      sendState(true);

      if (input) {
        const handleInput = () => {
          setState({ text: input.value, submitted: false });
        };
        const handleKeyDown = (event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            setState({ text: input.value, submitted: true }, { immediate: true });
            window.setTimeout(() => {
              setState({ submitted: false });
            }, debounceMs);
          }
        };
        const handleBlur = () => {
          setState({ submitted: false });
        };

        input.addEventListener("input", handleInput);
        input.addEventListener("keydown", handleKeyDown);
        input.addEventListener("blur", handleBlur);

        cleanupFns.push(() => {
          input.removeEventListener("input", handleInput);
          input.removeEventListener("keydown", handleKeyDown);
          input.removeEventListener("blur", handleBlur);
        });

        controller.updateInputValue = (value) => {
          const nextValue = coerceString(value, "");
          if (input.value !== nextValue) {
            input.value = nextValue;
          }
          state.text = coerceString(input.value, nextValue);
        };
      }

      

      let perLineSegs = sanitizeSegments(options.serializedSegments);
      controller.getLineCount = () => perLineSegs.length;
      if (!perLineSegs.length) {
        perLineSegs = fallbackSegmentsFromLines(normalisedLines, suffix);
      }

      let perLineSegmentHtml = [];
      let perLineHtml = [];
      let finalHtml = "";
      let totalLines = 0;

      const rebuildHtmlCaches = () => {
        perLineSegmentHtml = perLineSegs.map((segments) =>
          segments.map((segment) =>
            segment && segment.c
              ? `<span class="${segment.c}">${escapeHtml(segment.t)}</span>`
              : escapeHtml(segment && segment.t)
          )
        );
        perLineHtml = perLineSegmentHtml.map((parts) => parts.join(""));
        finalHtml =
          typeof payload.fullHtml === "string" && payload.fullHtml
            ? payload.fullHtml
            : perLineHtml.join("");
        totalLines = perLineSegs.length;
      };

      rebuildHtmlCaches();

      let prefilledLineCount = 0;
      let doneHtmlParts = [];

      const renderDoneHtml = () => {
        if (doneHtmlParts.length) {
          pre.innerHTML = doneHtmlParts.join("");
        } else {
          pre.innerHTML = "";
        }
      };

      const recomputePrefill = () => {
        prefilledLineCount = Math.max(
          0,
          Math.min(totalLines, coerceNumber(payload.prefilledLineCount, 0))
        );
        doneHtmlParts =
          prefilledLineCount > 0
            ? perLineHtml.slice(0, prefilledLineCount)
            : [];
        renderDoneHtml();
      };

      recomputePrefill();

      const ensureMeasuredHeight = () => {
        const measurement = rootNode.cloneNode(true);
        measurement.removeAttribute("id");
        measurement.style.position = "absolute";
        measurement.style.visibility = "hidden";
        measurement.style.pointerEvents = "none";
        measurement.style.opacity = "0";
        measurement.style.left = "-9999px";
        measurement.style.top = "0";
        measurement.style.height = "auto";
        measurement.style.minHeight = "auto";
        measurement.style.maxHeight = "none";
        const width =
          rootNode.getBoundingClientRect().width ||
          rootNode.offsetWidth ||
          rootNode.clientWidth;
        if (width) {
          measurement.style.width = `${width}px`;
        }
        const measurePre = measurement.querySelector(`.term-body-${suffix}`);
        if (measurePre) {
          measurePre.innerHTML = finalHtml;
        }
        const measureCaret = measurement.querySelector(`.caret-${suffix}`);
        if (measureCaret) {
          measureCaret.style.display = "none";
        }
        const measureWrap =
          measurement.querySelector(
            `.term-input-wrap-${suffix}:not([data-secondary="true"])`
          ) || measurement.querySelector(`.term-input-wrap-${suffix}`);
        const readyActive =
          rootNode.getAttribute("data-ready") === "true" && acceptKeystrokes;
        if (measureWrap) {
          measureWrap.setAttribute("data-active", readyActive ? "true" : "false");
          measureWrap.setAttribute("aria-hidden", readyActive ? "false" : "true");
        }
        const measureInput =
          measurement.querySelector(
            `.term-input-${suffix}:not([data-secondary="true"])`
          ) || measurement.querySelector(`.term-input-${suffix}`);
        if (measureInput) {
          measureInput.disabled = !readyActive;
        }
        (rootNode.parentElement || document.body).appendChild(measurement);
        const height = measurement.scrollHeight;
        measurement.remove();
        if (height) {
          rootNode.style.minHeight = `${height}px`;
          rootNode.style.height = `${height}px`;
        }
        return height;
      };

      const initialHeight = ensureMeasuredHeight();
      notifyResize(initialHeight);

      const mediaQuery =
        typeof window.matchMedia === "function"
          ? window.matchMedia("(prefers-reduced-motion: reduce)")
          : null;
      const prefersReduced = Boolean(mediaQuery && mediaQuery.matches);

      let finished = false;
      const finalizeTerminal = () => {
        if (finished) {
          return;
        }
        finished = true;
        setState({ ready: true }, { immediate: true });
        const measured = ensureMeasuredHeight();
        notifyResize(measured);
      };

      if (
        prefersReduced ||
        coerceNumber(payload.speedType, 0) === 0 ||
        prefilledLineCount >= totalLines
      ) {
        pre.innerHTML = finalHtml;
        if (caret) {
          caret.style.display = "none";
        }
        finalizeTerminal();
        controller.destroy = () => {
          if (debounceHandle !== null) {
            window.clearTimeout(debounceHandle);
          }
          if (typingHandle !== null) {
            window.clearTimeout(typingHandle);
          }
          if (rafHandle !== null) {
            window.cancelAnimationFrame(rafHandle);
          }
          cleanupFns.forEach((fn) => fn());
        };
        return controller;
      }

      let typingConfig = options.typingConfig || {};
      let typeDelay = 0;
      let betweenLines = 0;

      const applyTypingConfig = (config) => {
        typingConfig = config && typeof config === "object" ? { ...config } : {};
        typeDelay = Math.max(
          0,
          coerceNumber(typingConfig.speedType, payload.speedType || 0)
        );
        betweenLines = Math.max(
          0,
          coerceNumber(typingConfig.pauseBetween, payload.pauseBetween || 0)
        );
      };

      applyTypingConfig(typingConfig);

      let activeNode = null;
      let lineIndex = prefilledLineCount;
      let segmentIndex = 0;
      let charIndex = 0;

      const syncDoneHtml = () => {
        renderDoneHtml();
      };

      const ensureActiveNode = () => {
        if (activeNode) {
          return activeNode;
        }
        const segments = perLineSegs[lineIndex] || [];
        const current = segments[segmentIndex];
        if (!current) {
          return null;
        }
        if (current.c) {
          const span = document.createElement("span");
          span.className = current.c;
          span.textContent = "";
          pre.appendChild(span);
          activeNode = span;
        } else {
          activeNode = document.createTextNode("");
          pre.appendChild(activeNode);
        }
        return activeNode;
      };

      const commitActiveSegment = () => {
        if (!activeNode) {
          return;
        }
        if (activeNode.parentNode === pre) {
          pre.removeChild(activeNode);
        }
        const segmentHtml =
          (perLineSegmentHtml[lineIndex] || [])[segmentIndex] || "";
        doneHtmlParts.push(segmentHtml);
        activeNode = null;
        syncDoneHtml();
      };

      const cancelTypingTimers = () => {
        if (typingHandle !== null) {
          window.clearTimeout(typingHandle);
          typingHandle = null;
        }
        if (rafHandle !== null) {
          window.cancelAnimationFrame(rafHandle);
          rafHandle = null;
        }
      };

      const resetTypingIndices = (startLine) => {
        cancelTypingTimers();
        if (activeNode && activeNode.parentNode === pre) {
          pre.removeChild(activeNode);
        }
        activeNode = null;
        lineIndex = Math.max(
          0,
          typeof startLine === "number" ? startLine : prefilledLineCount
        );
        segmentIndex = 0;
        charIndex = 0;
      };

      const scheduleStep = (delay) => {
        if (typingHandle !== null) {
          window.clearTimeout(typingHandle);
        }
        typingHandle = window.setTimeout(step, Math.max(0, delay));
      };

      const step = () => {
        if (lineIndex >= perLineSegs.length) {
          syncDoneHtml();
          if (caret) {
            caret.style.display = "none";
          }
          finalizeTerminal();
          return;
        }

        const segments = perLineSegs[lineIndex] || [];
        const current = segments[segmentIndex];

        if (!current) {
          segmentIndex = 0;
          lineIndex += 1;
          scheduleStep(betweenLines);
          return;
        }

        const target = coerceString(current.t, "");
        if (charIndex < target.length) {
          const node = ensureActiveNode();
          if (node) {
            const char = target.charAt(charIndex);
            node.textContent = `${node.textContent || ""}${char}`;
          }
          charIndex += 1;
          scheduleStep(typeDelay);
          return;
        }

        commitActiveSegment();
        segmentIndex += 1;
        charIndex = 0;

        if (segmentIndex >= segments.length) {
          segmentIndex = 0;
          lineIndex += 1;
          scheduleStep(betweenLines);
          return;
        }

        scheduleStep(typeDelay);
      };

      const beginTyping = (startLine) => {
        resetTypingIndices(startLine);
        finished = false;
        if (caret) {
          caret.style.display = payload.showCaret === false ? "none" : "inline-block";
        }
        rafHandle = window.requestAnimationFrame(step);
      };

      const markReadyFalse = () => {
        wasReady = false;
        setState({ ready: false }, { immediate: true });
      };

      const completeImmediately = () => {
        cancelTypingTimers();
        doneHtmlParts = perLineHtml.slice();
        renderDoneHtml();
        if (caret) {
          caret.style.display = "none";
        }
        lineIndex = perLineSegs.length;
        segmentIndex = 0;
        charIndex = 0;
        finalizeTerminal();
      };

      const prepareUpdate = (updateOptions) => {
        if (updateOptions && typeof updateOptions === "object") {
          if (Object.prototype.hasOwnProperty.call(updateOptions, "payload")) {
            syncPayload(updateOptions.payload);
          }
          if (
            Object.prototype.hasOwnProperty.call(
              updateOptions,
              "acceptKeystrokes"
            )
          ) {
            controller.updateAcceptKeystrokes(updateOptions.acceptKeystrokes);
          }
          if (
            updateOptions.typingConfig &&
            typeof updateOptions.typingConfig === "object"
          ) {
            applyTypingConfig(updateOptions.typingConfig);
          } else {
            applyTypingConfig(typingConfig);
          }
        } else {
          applyTypingConfig(typingConfig);
        }
      };

      const toSegmentHtml = (segments) =>
        segments.map((segment) =>
          segment && segment.c
            ? `<span class="${segment.c}">${escapeHtml(segment.t)}</span>`
            : escapeHtml(segment && segment.t)
        );

      controller.replaceSerializedSegments = (rawSegments, updateOptions) => {
        prepareUpdate(updateOptions);
        let nextSegments = sanitizeSegments(rawSegments);
        if (!nextSegments.length) {
          nextSegments = fallbackSegmentsFromLines(normalisedLines, suffix);
        }
        perLineSegs = nextSegments;
        rebuildHtmlCaches();
        recomputePrefill();
        resetTypingIndices(prefilledLineCount);
        const measured = ensureMeasuredHeight();
        notifyResize(measured);
        if (
          prefersReduced ||
          coerceNumber(payload.speedType, 0) === 0 ||
          prefilledLineCount >= totalLines
        ) {
          completeImmediately();
          return;
        }
        markReadyFalse();
        beginTyping(prefilledLineCount);
      };

      controller.appendSerializedSegments = (rawSegments, updateOptions) => {
        prepareUpdate(updateOptions);
        const newSegments = sanitizeSegments(rawSegments);
        if (!newSegments.length) {
          return;
        }
        const previousTotal = perLineSegs.length;
        newSegments.forEach((segments) => {
          perLineSegs.push(segments);
          const htmlParts = toSegmentHtml(segments);
          perLineSegmentHtml.push(htmlParts);
          perLineHtml.push(htmlParts.join(""));
        });
        totalLines = perLineSegs.length;
        finalHtml =
          typeof payload.fullHtml === "string" && payload.fullHtml
            ? payload.fullHtml
            : perLineHtml.join("");
        const measured = ensureMeasuredHeight();
        notifyResize(measured);
        if (prefersReduced || coerceNumber(payload.speedType, 0) === 0) {
          completeImmediately();
          return;
        }
        if (finished) {
          doneHtmlParts = perLineHtml.slice(0, previousTotal);
          renderDoneHtml();
          markReadyFalse();
          beginTyping(previousTotal);
        }
      };

      controller.updatePayloadOnly = (updateOptions) => {
        prepareUpdate(updateOptions);
        const measured = ensureMeasuredHeight();
        notifyResize(measured);
      };

      markReadyFalse();
      beginTyping(prefilledLineCount);

      controller.destroy = () => {
        finished = true;
        if (debounceHandle !== null) {
          window.clearTimeout(debounceHandle);
        }
        if (typingHandle !== null) {
          window.clearTimeout(typingHandle);
        }
        if (rafHandle !== null) {
          window.cancelAnimationFrame(rafHandle);
        }
        cleanupFns.forEach((fn) => fn());
      };

      return controller;
    };

    const render = (args) => {
      const payload =
        args.payload && typeof args.payload === "object"
          ? { ...args.payload }
          : {};
      const serializedLines = Array.isArray(args.serializedLines)
        ? args.serializedLines
        : [];
      const typingConfig =
        args.typingConfig && typeof args.typingConfig === "object"
          ? args.typingConfig
          : {};
      const markup = coerceString(args.markup, "");
      const domId = coerceString(payload.domId, "");
      const nextInputValue = coerceString(payload.inputValue, "");
      const serializedSignature = stableStringify(serializedLines);

      if (!domId) {
        if (activeController && typeof activeController.destroy === "function") {
          activeController.destroy();
        }
        activeController = null;
        mountedDomId = null;
        mountedMarkupKey = null;
        mountedMarkupSignature = null;
        lastSerializedSignature = null;
        pushState(defaultState());
        notifyResize(document.body ? document.body.scrollHeight : 0);
        return;
      }

      const updateOptions = {
        payload,
        typingConfig,
        acceptKeystrokes: args.acceptKeystrokes,
      };

      const mountMarkupIfNeeded = () => {
        const domChanged = mountedMarkupKey !== domId;
        const signatureChanged = mountedMarkupSignature !== markup;
        if (!domChanged && !signatureChanged) {
          return false;
        }
        if (activeController && typeof activeController.destroy === "function") {
          activeController.destroy();
        }
        activeController = null;
        mountedDomId = null;
        root.innerHTML = markup;
        mountedMarkupKey = domId;
        mountedMarkupSignature = markup;
        return true;
      };

      const createController = () => {
        mountMarkupIfNeeded();
        const terminalRoot = document.getElementById(domId);
        if (!terminalRoot) {
          pushState(defaultState());
          notifyResize(document.body ? document.body.scrollHeight : 0);
          activeController = null;
          mountedDomId = null;
          mountedMarkupKey = null;
          mountedMarkupSignature = null;
          return;
        }
        const controller = initializeTerminal(terminalRoot, {
          payload,
          serializedSegments: serializedLines,
          typingConfig,
          acceptKeystrokes: args.acceptKeystrokes,
        });
        if (!controller) {
          pushState(defaultState());
          notifyResize(document.body ? document.body.scrollHeight : 0);
          activeController = null;
          mountedDomId = null;
          mountedMarkupKey = null;
          mountedMarkupSignature = null;
          return;
        }
        activeController = controller;
        mountedDomId = domId;
        lastSerializedSignature = serializedSignature;
      };

      const markupRemounted = mountMarkupIfNeeded();

      if (!activeController || mountedDomId !== domId) {
        if (activeController && typeof activeController.destroy === "function") {
          activeController.destroy();
        }
        activeController = null;
        createController();
      } else if (markupRemounted) {
        createController();
      } else if (activeController) {
          if (
            typeof activeController.updateInputValue === "function" &&
            nextInputValue !== lastInputValue
          ) {
            activeController.updateInputValue(nextInputValue);
          }

          const currentCount =
            typeof activeController.getLineCount === "function"
              ? activeController.getLineCount()
              : 0;
          const nextCount = serializedLines.length;

          if (nextCount > currentCount) {
            const delta = serializedLines.slice(currentCount);
            activeController.appendSerializedSegments(delta, updateOptions);
          } else if (nextCount < currentCount) {
            activeController.replaceSerializedSegments(
              serializedLines,
              updateOptions
            );
          } else {
            if (
              serializedSignature &&
              lastSerializedSignature &&
              serializedSignature !== lastSerializedSignature
            ) {
              activeController.replaceSerializedSegments(
                serializedLines,
                updateOptions
              );
            } else if (
              typeof activeController.updatePayloadOnly === "function"
            ) {
              activeController.updatePayloadOnly(updateOptions);
            }
          }
          lastSerializedSignature = serializedSignature;
        }
      }

      if (!activeController) {
        mountedMarkupKey = null;
        mountedDomId = null;
        mountedMarkupSignature = null;
      }

      lastInputValue = nextInputValue;
    };

    const onRender = (event) => {
      const detail = event.detail || {};
      const args = detail.args || {};
      render(args);
    };

    if (canListen && renderEvent) {
      Streamlit.events.addEventListener(renderEvent, onRender);
    } else if (initialArgs) {
      render(initialArgs);
    } else {
      document.addEventListener("DOMContentLoaded", () => {
        render({});
      });
    }

    if (canSetReady) {
      Streamlit.setComponentReady();
    }
    notifyResize(document.body ? document.body.scrollHeight : 0);
  };

  const startWhenStreamlitReady = () => {
    const Streamlit = resolveStreamlit();
    if (!Streamlit) {
      window.setTimeout(startWhenStreamlitReady, 50);
      return;
    }
    bootstrap(Streamlit, null);
  };

  startWhenStreamlitReady();
})();
