import { createInputController } from "./inputController.js";
import { createResizeManager } from "./resizeManager.js";
import {
  createStreamlitAdapter,
  resolveStreamlit,
} from "./streamlitAdapter.js";
import { createTypingAnimator } from "./typingAnimator.js";
import {
  coerceNumber,
  coerceString,
  defaultState,
  sanitizeDeltaPayload,
  stableStringify,
} from "./utils.js";

const root = document.getElementById("root");

if (!root) {
  // The root node is injected via index.html; bail gracefully if it is missing.
  // This mirrors the legacy behaviour where the script no-oped when the root
  // was absent.
  console.warn("Terminal root element not found");
}

let streamlitAdapter = null;
let activeController = null;
let mountedDomId = null;
let lastInputValue = null;
let lastSerializedSignature = null;
let mountedMarkupKey = null;
let mountedMarkupSignature = null;

const pushState = (state) => {
  if (streamlitAdapter) {
    streamlitAdapter.pushState(state);
  }
};

const notifyResize = (height) => {
  if (streamlitAdapter) {
    streamlitAdapter.notifyResize(height);
  }
};

const resetState = () => {
  pushState(defaultState());
};

const initializeTerminal = (rootNode, options = {}) => {
  if (!streamlitAdapter) {
    return null;
  }

  let payload = options.payload || {};
  const suffix = coerceString(payload.suffix, "");
  if (!suffix) {
    resetState();
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

  const controller = {
    domId: coerceString(payload.domId, ""),
    suffix,
    destroy: () => {},
    updateInputValue: () => {},
    updateAcceptKeystrokes: () => {},
    replaceSerializedSegments: () => {},
    appendSerializedSegments: () => {},
    updatePayloadOnly: () => {},
    getLineCount: () => 0,
  };

  const acceptOverride = options.acceptKeystrokes;
  let acceptKeystrokes =
    typeof acceptOverride === "boolean"
      ? acceptOverride
      : Boolean(payload.acceptKeystrokes);
  const debounceMs = Math.max(0, coerceNumber(payload.debounceMs, 150));
  let keepInputActive = Boolean(payload.keepInput);

  const initialText = input
    ? coerceString(payload.inputValue, input.value)
    : coerceString(payload.inputValue, "");
  if (input) {
    input.value = initialText;
  }

  rootNode.style.minHeight = "";
  rootNode.style.height = "";

  const state = {
    text: input ? coerceString(input.value, initialText) : initialText,
    submitted: false,
    ready: false,
  };
  let wasReady = false;
  let debounceHandle = null;

  const resizeManager = createResizeManager(rootNode, notifyResize);
  const cleanupFns = [resizeManager.destroy];
  const scheduleResizeNotification = resizeManager.scheduleResizeNotification;

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
    if (!streamlitAdapter) {
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
    if (!streamlitAdapter) {
      return;
    }
    if (immediate) {
      flushState();
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
        if (!wasReady && inputController) {
          wasReady = true;
          window.requestAnimationFrame(() => {
            inputController.focusOnReady();
          });
        }
      } else {
        input.disabled = true;
        input.setAttribute("aria-disabled", "true");
      }
    }
    scheduleResizeNotification();
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
      if (!state.ready) {
        wasReady = false;
      }
      syncReadyUi();
    }
    sendState(immediate);
  };

  const inputController = createInputController({
    input,
    inputWrap,
    suffix,
    initialText,
    setState,
    coerceString,
    debounceMs,
  });

  if (inputController) {
    cleanupFns.push(inputController.destroy);
    controller.updateInputValue = (value) => {
      inputController.updateInputValue(value);
      state.text = coerceString(input ? input.value : value, "");
    };
  }

  const syncPayload = (nextPayload) => {
    payload = nextPayload && typeof nextPayload === "object" ? nextPayload : {};
    keepInputActive = Boolean(payload.keepInput);
    const placeholderText = coerceString(payload.placeholder, "");
    const inputLabel = coerceString(payload.inputLabel, "");
    const inputAriaLabel = coerceString(payload.inputAriaLabel, "");
    const inputHint = coerceString(payload.inputHint, "");
    const inputId = coerceString(payload.inputId, "");
    const inputHintId = coerceString(payload.inputHintId, "");
    if (inputController) {
      inputController.syncAria({
        placeholderText,
        inputAriaLabel,
        inputHintId,
        inputHint,
        inputId,
        inputLabel,
      });
    } else if (input) {
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
      if (inputWrap) {
        const labelEl = inputWrap.querySelector(
          `label.sr-only-${suffix}[for="${inputId}"]`
        );
        if (labelEl && labelEl.textContent !== inputLabel) {
          labelEl.textContent = inputLabel;
        }
      }
      if (inputHintId) {
        const hintEl = document.getElementById(inputHintId);
        if (hintEl && hintEl.textContent !== inputHint) {
          hintEl.textContent = inputHint;
        }
      }
    }
  };

  syncPayload(payload);

  if (!pre) {
    resetState();
    return null;
  }

  const markReadyFalse = (options = {}) => {
    const preserveFocus = Boolean(
      options && typeof options === "object" && options.preserveFocus
    );
    if (preserveFocus) {
      return;
    }
    wasReady = false;
    setState({ ready: false }, { immediate: true });
  };

  const getPayload = () => payload;
  const setPayload = (nextPayload) => {
    syncPayload(nextPayload);
  };

  const prefersReducedMotion = (() => {
    if (typeof window.matchMedia === "function") {
      const query = window.matchMedia("(prefers-reduced-motion: reduce)");
      return Boolean(query && query.matches);
    }
    return false;
  })();

  const typing = createTypingAnimator({
    controller,
    getPayload,
    setPayload,
    serializedSegments: options.serializedSegments || [],
    typingConfig: options.typingConfig,
    scheduleResizeNotification,
    setReady: (ready, opts) => setState({ ready }, opts),
    markReadyFalse,
    keepInputActiveRef: () => keepInputActive,
    pre,
    caret,
    prefersReducedMotion,
    registerCleanup: (fn) => {
      if (typeof fn === "function") {
        cleanupFns.push(fn);
      }
    },
  });

  const controllerDestroy = typing.controller.destroy;

  const finalController = {
    ...typing.controller,
    destroy: () => {
      if (typeof controllerDestroy === "function") {
        controllerDestroy();
      } else {
        typing.destroy();
      }
      if (debounceHandle !== null) {
        window.clearTimeout(debounceHandle);
      }
      cleanupFns.forEach((fn) => fn());
    },
  };

  syncReadyUi();
  sendState(true);
  typing.start();

  return finalController;
};

let renderImpl = () => {};

const render = (args) => {
  renderImpl(args);
};

const bootstrap = (Streamlit, initialArgs) => {
  streamlitAdapter = createStreamlitAdapter(Streamlit);

  renderImpl = (args) => {
    if (!root) {
      resetState();
      return;
    }
    const payload =
      args.payload && typeof args.payload === "object"
        ? { ...args.payload }
        : {};
    const serializedLines = Array.isArray(args.serializedLines)
      ? args.serializedLines
      : null;
    const typingConfig =
      args.typingConfig && typeof args.typingConfig === "object"
        ? args.typingConfig
        : {};
    const markup = coerceString(args.markup, "");
    const domId = coerceString(payload.domId, "");
    const nextInputValue = coerceString(payload.inputValue, "");
    const serializedSignature =
      serializedLines !== null ? stableStringify(serializedLines) : null;
    const rawDelta =
      args.serializedDelta ||
      (payload && typeof payload.lineDelta === "object"
        ? payload.lineDelta
        : null);
    const lineDelta = sanitizeDeltaPayload(rawDelta);

    if (!domId) {
      if (activeController && typeof activeController.destroy === "function") {
        activeController.destroy();
      }
      activeController = null;
      mountedDomId = null;
      mountedMarkupKey = null;
      mountedMarkupSignature = null;
      lastSerializedSignature = null;
      resetState();
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
        resetState();
        notifyResize(document.body ? document.body.scrollHeight : 0);
        activeController = null;
        mountedDomId = null;
        mountedMarkupKey = null;
        mountedMarkupSignature = null;
        return;
      }
      const controller = initializeTerminal(terminalRoot, {
        payload,
        serializedSegments: serializedLines || [],
        typingConfig,
        acceptKeystrokes: args.acceptKeystrokes,
      });
      if (!controller) {
        resetState();
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

      const handleDeltaUpdate = () => {
        if (!lineDelta || typeof lineDelta.action !== "string") {
          return false;
        }
        const action = lineDelta.action;
        if (action === "append") {
          const segments = Array.isArray(lineDelta.segments)
            ? lineDelta.segments
            : [];
          if (!segments.length) {
            return false;
          }
          activeController.appendSerializedSegments(segments, updateOptions);
          return true;
        }
        if (action === "replace") {
          let segments = Array.isArray(lineDelta.segments)
            ? lineDelta.segments
            : null;
          if (!segments && serializedLines !== null) {
            segments = serializedLines;
          }
          if (!segments) {
            return false;
          }
          activeController.replaceSerializedSegments(segments, updateOptions);
          return true;
        }
        if (action === "none") {
          if (typeof activeController.updatePayloadOnly === "function") {
            activeController.updatePayloadOnly(updateOptions);
          }
          return true;
        }
        return false;
      };

      const deltaHandled = handleDeltaUpdate();

      if (!deltaHandled) {
        if (serializedLines !== null) {
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
          } else if (
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
        } else if (typeof activeController.updatePayloadOnly === "function") {
          activeController.updatePayloadOnly(updateOptions);
        }
      }

      if (deltaHandled) {
        if (serializedSignature !== null) {
          lastSerializedSignature = serializedSignature;
        } else if (lineDelta && lineDelta.action !== "none") {
          lastSerializedSignature = null;
        }
      } else {
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

  streamlitAdapter.registerRenderHandler(renderImpl, initialArgs);
  streamlitAdapter.setComponentReady();
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

export { initializeTerminal, render };
