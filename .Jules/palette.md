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

## 2024-05-23 - Accessibility of Custom Meters
**Learning:** Custom visualization components like progress bars or meters (often built with `div`s) are invisible to screen readers unless explicitly annotated with `role="progressbar"` (or `meter`) and proper ARIA values (`aria-valuenow`, `aria-valuemin`, `aria-valuemax`).
**Action:** Always check custom "bar" components for these attributes. When auditing, if you see a `div` representing a value, it likely needs ARIA.

## 2025-05-23 - Preserving Aesthetic during Refactor
**Learning:** When refactoring legacy components with heavy inline styles to Tailwind, particularly in themed applications (e.g., "Cyber"), be careful to map exact colors (like neon hex codes) to arbitrary values (e.g., `text-[#00f3ff]`) if they don't exist in the theme tokens, to preserve the specific aesthetic.
**Action:** Use arbitrary values for unique brand colors if not present in the design system to avoid visual regression.

## 2026-01-11 - Interactive List Accessibility
**Learning:** Scrollable lists of clickable items (like news feeds) are often implemented as `div`s with `onClick`, blocking keyboard access. Converting them to `<button>` elements inside a container with `tabIndex="0"` and `role="region"` restores full accessibility without compromising layout.
**Action:** Always implement clickable list items as buttons and ensure their container is keyboard-scrollable.

## 2026-01-12 - Identifying Debt via Inline Styles
**Learning:** Components using extensive inline `style` attributes often indicate legacy code or prototypes that lack both design system consistency and accessibility features. These are high-yield targets for refactoring.
**Action:** Use `grep "style={{"` to find and prioritize components that need both visual and accessible upgrades.

## 2026-02-15 - Micro-Interactions for Long Calculations
**Learning:** In data-heavy applications, users expect immediate feedback even for operations that take < 2 seconds. A simple spinner or "Analyzing..." text prevents rage clicks and reassures the user the system is working.
**Action:** Always add a loading state to "Run Analysis" or "Calculate" buttons, even if the backend is fast.

## 2026-05-24 - Terminal Command History Expectations
**Learning:** Users interacting with a "Terminal" UI component bring strong mental models from real CLIs, specifically expecting Up/Down arrow keys to cycle command history. Without this, the component feels "broken" or "fake" rather than just simple.
**Action:** When building terminal-like inputs, always implement history state and Up/Down navigation.

## 2026-05-24 - Associating Labels with Meters
**Learning:** Custom meters (like CPU usage) often have visual labels (e.g., "CPU: 65%") that are separate elements. Using `aria-labelledby` on the progressbar element pointing to the label's ID creates a semantic relationship that screen readers respect.
**Action:** Always use `id` and `aria-labelledby` to link custom controls to their visual text labels.

## 2026-07-18 - Label Association in Legacy Forms
**Learning:** Legacy form components often use implicit layouts (e.g., `div` wrapping `label` and `input`) without explicit `htmlFor`/`id` association, breaking accessibility.
**Action:** When refactoring, always add unique `id`s to inputs and `htmlFor` to labels, even if visual layout seems "obvious".

## 2027-02-27 - Button Loading State Feedback
**Learning:** When implementing loading states on buttons, simply disabling the button is insufficient. Users need explicit feedback (spinner + text change) to confirm the action was registered, especially for "simulation" type actions that imply a process.
**Action:** Use a conditional render inside the button to swap the action icon with a spinner and update the label text.

## 2025-05-27 - Styled Keyboard Shortcuts in Inputs
**Learning:** Embedding shortcut hints (like `[CTRL+K]`) directly in placeholder text lowers legibility and lacks visual hierarchy. Using an absolutely positioned, styled `<kbd>` element inside the input container reinforces the "Cyberpunk" aesthetic while keeping the input text clean.
**Action:** Extract shortcut hints from placeholders into distinct `<kbd>` components positioned at the right edge of the input.

## 2027-02-27 - Icon Affordance in Search
**Learning:** Search inputs often lack visual affordance when they are just plain text boxes. Adding a standard "Search" icon (magnifying glass) inside the input not only clarifies intent but also fits standard user expectations.
**Action:** When adding search inputs, always include a visual icon and ensure accessibility with `aria-label` or `aria-labelledby`.
