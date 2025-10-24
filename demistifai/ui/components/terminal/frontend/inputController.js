export const createInputController = ({
  input,
  inputWrap,
  suffix,
  initialText,
  setState,
  coerceString,
  debounceMs,
}) => {
  if (!input) {
    return null;
  }

  if (typeof initialText === "string" && input.value !== initialText) {
    input.value = initialText;
  }

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

  const destroy = () => {
    input.removeEventListener("input", handleInput);
    input.removeEventListener("keydown", handleKeyDown);
    input.removeEventListener("blur", handleBlur);
  };

  const updateInputValue = (value) => {
    const nextValue = coerceString(value, "");
    if (input.value !== nextValue) {
      input.value = nextValue;
    }
  };

  const focusOnReady = () => {
    try {
      input.focus({ preventScroll: true });
    } catch (_err) {
      input.focus();
    }
  };

  const syncAria = ({
    placeholderText,
    inputAriaLabel,
    inputHintId,
    inputHint,
    inputId,
    inputLabel,
  }) => {
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
    if (inputWrap && suffix) {
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
  };

  return {
    destroy,
    updateInputValue,
    focusOnReady,
    syncAria,
  };
};
