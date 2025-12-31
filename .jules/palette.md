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

## 2024-05-22 - Combobox Accessibility
**Learning:** Custom search dropdowns (comboboxes) are often completely inaccessible to keyboard users. Adding `aria-activedescendant` combined with `ArrowUp`/`ArrowDown` handlers allows screen readers to announce the currently selected item without moving focus, maintaining the typing context.
**Action:** Always implement `role="combobox"` pattern with `aria-activedescendant` for autocomplete/search inputs.

## 2025-12-18 - Empty States in Dashboards
**Learning:** Dashboards often initialize with empty arrays. Without a dedicated empty state, the UI looks broken or invisible to the user.
**Action:** Always check array length and render a helpful, styled 'empty' component with `role="status"`.

## 2025-05-20 - Accessible Tabs Implementation
**Learning:** Native `button` elements in a row are not sufficient for a "Tabs" pattern. Accessibility standards (WAI-ARIA) require `role="tablist"`/`tab` and *arrow key* navigation, not just Tab key (which should move focus *out* of the tab list).
**Action:** When implementing tabs, always handle `onKeyDown` for ArrowLeft/Right/Home/End and manage `tabIndex` so only the active tab is focusable (or all are, depending on roving tabindex preference).

## 2025-05-21 - Generated Form Accessibility
**Learning:** When generating forms from a schema (like JSON schema), it's easy to forget unique IDs for inputs, breaking label association. Using a combination of parent ID and field name creates robust, unique IDs.
**Action:** Always generate deterministic IDs for dynamic inputs and explicitly link them with `htmlFor` attributes on their labels.

## 2025-05-21 - Raw Data Editing Pattern
**Learning:** When allowing users to edit raw JSON data that drives an underlying object state, do not bind the input directly to the object stringification. This causes a loop where invalid JSON prevents updates or resets the cursor. Use a separate string state for the editor, and parse/commit it to the object state only on valid submit/blur.
**Action:** Use separate 'draft' string state for raw editors.

## 2025-05-22 - Interactive Divs vs Buttons
**Learning:** Using `div` with `onClick` for main navigation controls (like mode switchers) breaks keyboard accessibility completely. It's a common pattern in "custom" UIs but excludes keyboard users.
**Action:** Always use semantic `<button>` elements for actions, even if styling requires stripping default button styles (`background: none; border: none`).

## 2025-05-22 - Testing Interactive Components
**Learning:** For components with complex keyboard interactions (like terminals or comboboxes), relying on manual verification is risky. Using `@testing-library/react`'s `fireEvent.keyDown` allows for robust, automated testing of these interactions.
**Action:** Always write unit tests for keyboard navigation logic.
## 2025-12-24 - Loading State Accessibility
**Learning:** Stylistic loading animations (like CSS spinners) are invisible to screen readers. Adding `role="status"` and a visually hidden text span ensures assistive technology users know content is loading.
**Action:** Always pair visual spinners with invisible descriptive text and proper ARIA live regions.

## 2025-12-24 - Verify Rendered Components
**Learning:** In a codebase with legacy or duplicate files (e.g., `Header.js` vs `GlobalNav.tsx`), always verify which component is actually being rendered by the application (via `App.tsx` or `Layout.tsx`) before applying fixes. Screenshots and entry point analysis are crucial.
**Action:** Trace the component tree from `App.tsx` or `index.js` before assuming a file is active.

## 2025-12-31 - [Tailwind Reset Hides Focus Rings]
**Learning:** Tailwind's `appearance-none` utility removes default browser styling from form inputs, which inadvertently strips the native accessibility focus ring. This makes keyboard navigation impossible for these elements.
**Action:** When using `appearance-none` on interactive elements, always explicitly re-add focus styles using utilities like `focus-visible:ring-2` to ensure accessibility without cluttering the UI for mouse users.
