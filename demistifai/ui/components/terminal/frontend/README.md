# Terminal frontend structure

The terminal component is now composed from small modules so that behaviours can be updated without wading through a single monolithic script.

- `terminal.js` remains the browser entry point and exposes the same `initializeTerminal` and `render` APIs. It wires the modules together and manages the component lifecycle that Streamlit expects.
- `streamlitAdapter.js` hides Streamlit-specific concerns (state updates, resize notifications, render events) so helpers can interact with a simple interface.
- `resizeManager.js` watches the terminal container and triggers resize callbacks. Tweaks to height measurement or observer strategies live here.
- `inputController.js` centralises DOM wiring for the user input field (events, ARIA attributes, focus management).
- `typingAnimator.js` owns the incremental rendering/typing animation, keeping payload transforms and timing logic in one place.
- `utils.js` contains the shared coercion/sanitisation helpers used across the modules.

When adjusting typing cadence, resize heuristics, or input handling, look to the corresponding module before touching `terminal.js`. This keeps the bootstrap focused on composition and reduces the surface area for regressions.
