(function () {
  const parentWindow = window.parent;
  if (!parentWindow || parentWindow === window) {
    return;
  }

  if (window.Streamlit) {
    if (!window.streamlit_component_lib) {
      window.streamlit_component_lib = { Streamlit: window.Streamlit };
    } else if (!window.streamlit_component_lib.Streamlit) {
      window.streamlit_component_lib.Streamlit = window.Streamlit;
    }
    return;
  }

  const MESSAGE_TYPES = {
    RENDER: "streamlit:render",
    COMPONENT_READY: "streamlit:componentReady",
    SET_COMPONENT_VALUE: "streamlit:setComponentValue",
    SET_FRAME_HEIGHT: "streamlit:setFrameHeight",
  };

  const API_VERSION = 1;
  const eventTarget = document.createDocumentFragment();

  const postMessage = (type, payload) => {
    try {
      parentWindow.postMessage({ type, ...payload }, "*");
    } catch (error) {
      console.error("Streamlit shim failed to post message", error);
    }
  };

  const events = {
    addEventListener(type, handler, options) {
      eventTarget.addEventListener(type, handler, options);
    },
    removeEventListener(type, handler, options) {
      eventTarget.removeEventListener(type, handler, options);
    },
    dispatchEvent(event) {
      eventTarget.dispatchEvent(event);
    },
  };

  const Streamlit = {
    setComponentReady() {
      postMessage(MESSAGE_TYPES.COMPONENT_READY, { apiVersion: API_VERSION });
    },
    setComponentValue(value, options) {
      const payload = { value };
      const dataType = options && typeof options === "object" ? options.dataType : null;
      if (typeof dataType === "string" && dataType) {
        payload.dataType = dataType;
      } else {
        payload.dataType = "json";
      }
      postMessage(MESSAGE_TYPES.SET_COMPONENT_VALUE, payload);
    },
    setFrameHeight(height) {
      postMessage(MESSAGE_TYPES.SET_FRAME_HEIGHT, { height });
    },
    RENDER_EVENT: MESSAGE_TYPES.RENDER,
    events,
  };

  window.Streamlit = Streamlit;
  window.streamlit_component_lib = window.streamlit_component_lib || {};
  window.streamlit_component_lib.Streamlit = Streamlit;

  window.addEventListener("message", (event) => {
    const data = event.data;
    if (!data || data.type !== MESSAGE_TYPES.RENDER) {
      return;
    }
    const renderEvent = new CustomEvent(MESSAGE_TYPES.RENDER, { detail: data });
    events.dispatchEvent(renderEvent);
  });
})();
