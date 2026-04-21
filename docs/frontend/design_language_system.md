# Design Language System (DLS)

The project adheres to a distinct aesthetic guideline dubbed "Bloomberg Terminal meets Cyberpunk."

## Core Principles
1. **Dark Mode First**: The primary background is always a deep dark shade (e.g., `#030712` to `#0f172a`).
2. **Neon Accents**: Interactive elements, critical status indicators, and headers use vibrant neon colors.
   - *Cyan* (`#06b6d4`): Neutral/Processing/Links
   - *Red/Rose* (`#ef4444` / `#f43f5e`): Stress/Alerts/Divergence
   - *Amber* (`#f59e0b`): Warnings/Euphoria
   - *Green* (`#10b981`): Stable/Profitable
3. **Typography**: High density data presentation.
   - Headers: `Oswald`, `Inter` (often uppercase and tracking-tighter).
   - Data/Terminals: `Fira Code`, `ui-monospace`.
   - Body: `Inter`.
4. **UI Paradigms**:
   - Heavy use of `glass-card` components (backdrop blur with low-opacity borders).
   - Terminal-style "System Status" readout headers at the top of major views.
   - Scanlines and subtle animated gradients (e.g., `animate-fade-in`).
