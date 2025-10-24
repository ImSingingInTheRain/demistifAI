export const defaultState = () => ({ text: "", submitted: false, ready: false });

export const coerceString = (value, fallback = "") =>
  typeof value === "string" ? value : fallback;

export const coerceNumber = (value, fallback = 0) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

export const sanitizeClass = (value) => {
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

export const sanitizeSegments = (rawSegments) => {
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

export const sanitizeDeltaPayload = (rawDelta) => {
  if (!rawDelta || typeof rawDelta !== "object") {
    return null;
  }
  const action = coerceString(rawDelta.action, "").toLowerCase();
  if (!action) {
    return null;
  }
  const payload = { action };
  if (action === "append" || action === "replace") {
    const segments = sanitizeSegments(rawDelta.segments);
    if (segments.length) {
      payload.segments = segments;
    }
  }
  const prefilled = coerceNumber(rawDelta.prefilledLineCount, NaN);
  if (Number.isFinite(prefilled) && prefilled >= 0) {
    payload.prefilledLineCount = prefilled;
  }
  const total = coerceNumber(rawDelta.totalLineCount, NaN);
  if (Number.isFinite(total) && total >= 0) {
    payload.totalLineCount = total;
  }
  return payload;
};

export const buildEmptySegments = () => [];

export const ensureRenderableSegments = (segments) =>
  segments.length ? segments : buildEmptySegments();

export const escapeHtml = (text) =>
  coerceString(text, "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

export const stableStringify = (value) => {
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
