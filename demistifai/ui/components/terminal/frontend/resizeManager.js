export const createResizeManager = (rootNode, notifyResize) => {
  let resizeNotifyRaf = null;
  const cleanupFns = [];

  const measureHeight = () => {
    let height = 0;
    if (typeof rootNode.getBoundingClientRect === "function") {
      const rect = rootNode.getBoundingClientRect();
      if (rect && Number.isFinite(rect.height) && rect.height > 0) {
        height = rect.height;
      }
    }
    if (!height && rootNode.scrollHeight) {
      height = rootNode.scrollHeight;
    }
    if (!height && rootNode.offsetHeight) {
      height = rootNode.offsetHeight;
    }
    return height;
  };

  const scheduleResizeNotification = () => {
    if (resizeNotifyRaf !== null) {
      return;
    }
    resizeNotifyRaf = window.requestAnimationFrame(() => {
      resizeNotifyRaf = null;
      const measuredHeight = measureHeight();
      notifyResize(measuredHeight);
    });
  };

  if (typeof window.ResizeObserver === "function") {
    const resizeObserver = new window.ResizeObserver(() => {
      scheduleResizeNotification();
    });
    resizeObserver.observe(rootNode);
    cleanupFns.push(() => {
      resizeObserver.disconnect();
    });
  } else {
    const handleWindowResize = () => {
      scheduleResizeNotification();
    };
    window.addEventListener("resize", handleWindowResize);
    cleanupFns.push(() => {
      window.removeEventListener("resize", handleWindowResize);
    });
  }

  const destroy = () => {
    if (resizeNotifyRaf !== null) {
      window.cancelAnimationFrame(resizeNotifyRaf);
      resizeNotifyRaf = null;
    }
    cleanupFns.forEach((fn) => fn());
  };

  return { scheduleResizeNotification, destroy };
};
