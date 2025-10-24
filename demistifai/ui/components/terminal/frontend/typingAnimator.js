import {
  coerceNumber,
  coerceString,
  ensureRenderableSegments,
  escapeHtml,
  sanitizeSegments,
} from "./utils.js";

const toSegmentHtml = (segments) =>
  segments.map((segment) =>
    segment && segment.c
      ? `<span class="${segment.c}">${escapeHtml(segment.t)}</span>`
      : escapeHtml(segment && segment.t)
  );

const resolveTotalLineCount = (payload, fallbackLength) => {
  const reported = coerceNumber(payload.totalLineCount, NaN);
  if (Number.isFinite(reported) && reported >= 0) {
    return reported;
  }
  return fallbackLength;
};

const buildFinalHtml = (payload, perLineHtml) =>
  typeof payload.fullHtml === "string" && payload.fullHtml
    ? payload.fullHtml
    : perLineHtml.join("");

export const createTypingAnimator = ({
  controller,
  getPayload,
  setPayload,
  serializedSegments,
  typingConfig,
  scheduleResizeNotification,
  setReady,
  markReadyFalse,
  keepInputActiveRef,
  pre,
  caret,
  prefersReducedMotion,
  registerCleanup,
}) => {
  let payload = getPayload();
  let perLineSegs = ensureRenderableSegments(sanitizeSegments(serializedSegments));
  let perLineSegmentHtml = [];
  let perLineHtml = [];
  let finalHtml = "";
  let totalLines = 0;
  let prefilledLineCount = 0;
  let doneHtmlParts = [];

  let typingConfigState =
    typingConfig && typeof typingConfig === "object" ? { ...typingConfig } : {};
  let typeDelay = 0;
  let betweenLines = 0;

  const recalcTypingTimings = () => {
    const latestPayload = getPayload();
    typeDelay = Math.max(
      0,
      coerceNumber(typingConfigState.speedType, latestPayload.speedType || 0)
    );
    betweenLines = Math.max(
      0,
      coerceNumber(
        typingConfigState.pauseBetween,
        latestPayload.pauseBetween || 0
      )
    );
  };

  const rebuildHtmlCaches = () => {
    perLineSegmentHtml = perLineSegs.map((segments) => toSegmentHtml(segments));
    perLineHtml = perLineSegmentHtml.map((parts) => parts.join(""));
    finalHtml = buildFinalHtml(payload, perLineHtml);
    const resolvedTotal = resolveTotalLineCount(payload, perLineSegs.length);
    totalLines = Math.max(perLineSegs.length, resolvedTotal);
  };

  const recomputePrefill = () => {
    payload = getPayload();
    prefilledLineCount = Math.max(
      0,
      Math.min(totalLines, coerceNumber(payload.prefilledLineCount, 0))
    );
    doneHtmlParts =
      prefilledLineCount > 0 ? perLineHtml.slice(0, prefilledLineCount) : [];
  };

  const renderDoneHtml = () => {
    if (doneHtmlParts.length) {
      pre.innerHTML = doneHtmlParts.join("");
    } else {
      pre.innerHTML = "";
    }
    scheduleResizeNotification();
  };

  const refreshCaches = () => {
    rebuildHtmlCaches();
    recomputePrefill();
    renderDoneHtml();
  };

  rebuildHtmlCaches();
  recomputePrefill();
  renderDoneHtml();
  recalcTypingTimings();
  scheduleResizeNotification();

  const prefersReduced = Boolean(prefersReducedMotion);

  let activeNode = null;
  let lineIndex = prefilledLineCount;
  let segmentIndex = 0;
  let charIndex = 0;
  let finished = false;
  let typingHandle = null;
  let rafHandle = null;

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

  registerCleanup(() => {
    cancelTypingTimers();
  });

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
    const segmentHtml = (perLineSegmentHtml[lineIndex] || [])[segmentIndex] || "";
    doneHtmlParts.push(segmentHtml);
    activeNode = null;
    renderDoneHtml();
  };

  const resetTypingIndices = (startLine) => {
    cancelTypingTimers();
    if (activeNode && activeNode.parentNode === pre) {
      pre.removeChild(activeNode);
    }
    activeNode = null;
    lineIndex = Math.max(0, typeof startLine === "number" ? startLine : 0);
    segmentIndex = 0;
    charIndex = 0;
  };

  const scheduleStep = (delay) => {
    if (typingHandle !== null) {
      window.clearTimeout(typingHandle);
    }
    typingHandle = window.setTimeout(step, Math.max(0, delay));
  };

  const finalizeTerminal = () => {
    if (finished) {
      return;
    }
    finished = true;
    setReady(true, { immediate: true });
    scheduleResizeNotification();
  };

  const step = () => {
    if (lineIndex >= perLineSegs.length) {
      renderDoneHtml();
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
      const payloadNow = getPayload();
      caret.style.display = payloadNow.showCaret === false ? "none" : "inline-block";
    }
    rafHandle = window.requestAnimationFrame(step);
  };

  const shouldSkipTyping = () => {
    const payloadNow = getPayload();
    return (
      prefersReduced ||
      coerceNumber(payloadNow.speedType, 0) === 0 ||
      prefilledLineCount >= totalLines
    );
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

  if (shouldSkipTyping()) {
    completeImmediately();
  }

  const applyTypingConfig = (config) => {
    typingConfigState = config && typeof config === "object" ? { ...config } : {};
    recalcTypingTimings();
  };

  const evaluateNextPayloadState = (nextPayload) => {
    const safePayload = nextPayload && typeof nextPayload === "object" ? nextPayload : {};
    const resolvedTotal = resolveTotalLineCount(safePayload, perLineSegs.length);
    const nextTotal = Math.max(perLineSegs.length, resolvedTotal);
    const nextPrefill = Math.max(
      0,
      Math.min(nextTotal, coerceNumber(safePayload.prefilledLineCount, 0))
    );
    const nextFinalHtml = buildFinalHtml(safePayload, perLineHtml);
    return { nextTotal, nextPrefill, nextFinalHtml };
  };

  const prepareUpdate = (updateOptions, options = {}) => {
    const refreshOnPayload = Boolean(
      options && typeof options === "object" && options.refreshOnPayload
    );
    if (updateOptions && typeof updateOptions === "object") {
      if (Object.prototype.hasOwnProperty.call(updateOptions, "payload")) {
        const previousTotalLines = totalLines;
        const previousPrefill = prefilledLineCount;
        const previousFinalHtml = finalHtml;

        setPayload(updateOptions.payload);
        payload = getPayload() || {};

        let cacheState = null;
        let needsRefresh = refreshOnPayload;
        if (!needsRefresh) {
          cacheState = evaluateNextPayloadState(payload);
          if (
            cacheState.nextTotal !== previousTotalLines ||
            cacheState.nextFinalHtml !== previousFinalHtml
          ) {
            needsRefresh = true;
          }
        }

        if (needsRefresh) {
          refreshCaches();
        } else {
          if (!cacheState) {
            cacheState = evaluateNextPayloadState(payload);
          }
          if (cacheState.nextPrefill !== previousPrefill) {
            recomputePrefill();
            renderDoneHtml();
          }
        }
      }
      if (
        Object.prototype.hasOwnProperty.call(updateOptions, "acceptKeystrokes") &&
        typeof controller.updateAcceptKeystrokes === "function"
      ) {
        controller.updateAcceptKeystrokes(updateOptions.acceptKeystrokes);
      }
      if (
        updateOptions.typingConfig &&
        typeof updateOptions.typingConfig === "object"
      ) {
        applyTypingConfig(updateOptions.typingConfig);
      } else {
        recalcTypingTimings();
      }
    } else {
      recalcTypingTimings();
    }
  };

  const replaceSerializedSegments = (rawSegments, updateOptions) => {
    prepareUpdate(updateOptions, { refreshOnPayload: true });
    perLineSegs = ensureRenderableSegments(sanitizeSegments(rawSegments));
    rebuildHtmlCaches();
    recomputePrefill();
    renderDoneHtml();
    resetTypingIndices(prefilledLineCount);
    scheduleResizeNotification();
    if (shouldSkipTyping()) {
      completeImmediately();
      return;
    }
    markReadyFalse({ preserveFocus: keepInputActiveRef() });
    beginTyping(prefilledLineCount);
  };

  const appendSerializedSegments = (rawSegments, updateOptions) => {
    prepareUpdate(updateOptions, { refreshOnPayload: true });
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
    payload = getPayload();
    const resolvedTotal = resolveTotalLineCount(payload, perLineSegs.length);
    totalLines = Math.max(perLineSegs.length, resolvedTotal);
    finalHtml = buildFinalHtml(payload, perLineHtml);
    scheduleResizeNotification();
    if (shouldSkipTyping()) {
      completeImmediately();
      return;
    }
    if (finished) {
      doneHtmlParts = perLineHtml.slice(0, previousTotal);
      renderDoneHtml();
      markReadyFalse({ preserveFocus: keepInputActiveRef() });
      beginTyping(previousTotal);
    }
  };

  const updatePayloadOnly = (updateOptions) => {
    prepareUpdate(updateOptions);
    scheduleResizeNotification();
  };

  const destroyTyping = () => {
    finished = true;
    cancelTypingTimers();
  };

  const start = () => {
    if (shouldSkipTyping()) {
      completeImmediately();
      return;
    }
    markReadyFalse({ preserveFocus: keepInputActiveRef() });
    beginTyping(prefilledLineCount);
  };

  controller.getLineCount = () => perLineSegs.length;

  return {
    controller: {
      ...controller,
      replaceSerializedSegments,
      appendSerializedSegments,
      updatePayloadOnly,
      destroy: (...args) => {
        if (typeof controller.destroy === "function") {
          controller.destroy(...args);
        }
        destroyTyping();
      },
    },
    start,
    destroy: destroyTyping,
  };
};
