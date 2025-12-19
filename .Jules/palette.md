# Palette's Journal

## 2025-12-12 - Terminal Accessibility
**Learning:** Terminal-style interfaces often lack accessibility cues because they rely on visual "hacker" aesthetics. Adding `aria-live="polite"` to the output container is critical for screen readers to announce new command results.
**Action:** Always wrap dynamic log outputs in `role="log"` with `aria-live` and provide meaningful labels for command inputs.

## 2025-10-26 - Keyboard Shortcuts for Search
**Learning:** Users expect `Ctrl+K` or `Cmd+K` to focus global search bars, especially in developer-focused tools. Implementing this along with a visual `[CTRL+K]` hint creates a seamless experience.
**Action:** When adding search inputs, always pair a placeholder hint with a `keydown` listener for the shortcut.

## 2025-12-14 - Scrollable Regions Accessibility
**Learning:** `overflow: auto` regions are not keyboard accessible by default. Users cannot scroll them without a mouse unless they have `tabIndex="0"`.
**Action:** Always add `tabIndex="0"` and a visible focus indicator (`focus:ring`) to any scrollable container (like logs or terms).
