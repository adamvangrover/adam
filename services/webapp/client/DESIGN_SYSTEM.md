# Adam Platform Design System

This document formalizes the aesthetic guidelines for the Adam frontend, embodying a "Bloomberg Terminal meets Cyberpunk" style.

## Core Aesthetic
The overarching visual language balances institutional density (data-heavy, precise, utility-first) with cyberpunk accents (neon highlights, stark contrast, terminal-like interactivity).

## Color Palette
*   **Backgrounds**: True black (`#000000`) and deep slate (`#0F172A`).
*   **Text (Primary)**: High-contrast white (`#FFFFFF`) or bright off-white (`#F8FAFC`).
*   **Text (Secondary/Muted)**: Slate gray (`#64748B`).
*   **Accents**:
    *   **Neon Cyan (`#06B6D4`)**: Used for primary actions, active states, and optimistic data.
    *   **Neon Pink/Rose (`#F43F5E`)**: Used for warnings, stress indicators, and critical alerts.
    *   **Amber (`#F59E0B`)**: Used for divergent or cautious system status.

## Typography
*   **Monospace (Code/Data)**: Primary font for numbers, tickers, and code blocks (e.g., Fira Code, JetBrains Mono, or system monospace).
*   **Sans-Serif (UI/Text)**: Clean, highly legible sans-serif for general UI elements (e.g., Inter, Roboto).

## UI Paradigms
*   **Dense Data Displays**: Minimize padding in data tables to maximize information density.
*   **Glow Effects**: Use subtle CSS `text-shadow` or `box-shadow` to create a "neon glow" on critical status indicators.
*   **Terminal Elements**: Emulate command-line interfaces with visible cursors or blinking underscores for interactive text elements.
*   **3-Stage Workflow**: Most complex modules follow a strict hierarchy: Directory (Grid), Tearsheet (Summary Sidebar), and Drill-Down (Full-page Modal/View).
