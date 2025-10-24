import { defaultState } from "./utils.js";

export const resolveStreamlit = () =>
  window.Streamlit || window.streamlit_component_lib?.Streamlit || null;

export const createStreamlitAdapter = (Streamlit) => {
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

  const pushState = (state) => {
    if (!canSetValue) {
      return;
    }
    Streamlit.setComponentValue({ value: state });
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

  const setComponentReady = () => {
    if (canSetReady) {
      Streamlit.setComponentReady();
    }
  };

  const registerRenderHandler = (render, initialArgs) => {
    const onRender = (event) => {
      const detail = event.detail || {};
      render(detail.args || {});
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
  };

  const resetState = () => {
    pushState(defaultState());
  };

  return {
    canSetValue,
    canListen,
    renderEvent,
    pushState,
    notifyResize,
    setComponentReady,
    registerRenderHandler,
    resetState,
  };
};
