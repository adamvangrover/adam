# Prompt Alpha: Technical Specification

## 1. Overview
"Prompt Alpha" is a client-side-only "Bloomberg Terminal" for AI Prompts. It aggregates prompt feeds (e.g., Reddit), analyzes them locally using a proprietary "Alpha" scoring algorithm, and displays them in a high-frequency trading aesthetic.

## 2. Architecture
*   **Pattern:** Local-First / Zero-Backend.
*   **State Management:** Zustand with `localStorage` persistence.
*   **Data Fetching:** Client-side polling of public JSON endpoints.
*   **Scoring:** Deterministic JavaScript function executed in the main thread.
*   **Simulation:** Built-in "Synthetic Prompt Engine" to generate high-alpha test data when live feeds are unavailable.

## 3. Data Schema (`types/promptAlpha.ts`)

### `PromptObject`
The core entity representing a single prompt.
```typescript
interface PromptObject {
  id: string;
  title: string;
  content: string;
  source: string; // 'Reddit' | 'Simulation'
  timestamp: number;
  alphaScore: number; // 0-100
  metrics: {
    length: number;
    variableDensity: number;
    structuralKeywords: number;
  };
  tags: string[];
  isFavorite: boolean;
}
```

### `SourceFeed`
Configuration for an external data source.
```typescript
interface SourceFeed {
  id: string;
  url: string; // e.g., Reddit JSON
  isActive: boolean;
  refreshInterval: number;
}
```

## 4. Alpha Algorithm (`utils/alphaCalculator.ts`)
The `calculatePromptAlpha` function scores prompts based on three factors:

1.  **Complexity (Length):** Logarithmic scale favoring longer, detailed prompts up to ~2000 characters.
    *   *Rationale:* Short prompts usually lack "alpha".
2.  **Variable Density:** Counts occurrences of `{{placeholder}}` syntax.
    *   *Rationale:* Templated prompts are more reusable and engineered.
3.  **Structural Integrity:** Scans for keywords like "System:", "Step-by-step", "JSON".
    *   *Rationale:* Indicates advanced prompting techniques (Chain-of-Thought, Persona usage).

**Formula:**
`Alpha = Normalize( (LengthScore * 0.4) + (VariableScore * 0.3) + (StructureScore * 0.3) )`

## 5. Component Architecture

### `PromptAlpha.tsx` (Page)
The entry point. Manages the layout and initializes the `usePromptFeed` hook.
*   **Features:**
    *   **Live Stream / Vault View:** Toggle between incoming feed and saved favorites.
    *   **Simulation Mode:** Toggle to enable synthetic data generation.
    *   **Analytics:** Integrated Histogram of Alpha Scores.
    *   **Command Bar:** `NEWS <GO>`, `SYS <GO>` CLI navigation.

### `AlphaChart.tsx`
A `recharts` based histogram visualization showing the distribution of Alpha scores across the current buffer.

### `TickerTape.tsx`
A scrolling marquee at the top of the screen displaying the highest-alpha prompts detected in the last cycle.
*   *Visuals:* Green text on black background, CSS keyframe animation.

### `CommandBar.tsx`
A CLI-style input field for quick navigation between functions.

### `SystemMonitor.tsx`
Visualizes the health and load of core system agents (RiskAnalyst, ConsensusEngine, etc.) and memory allocation.

### `NewsWire.tsx`
A real-time scrolling news feed with flash updates and priority highlighting.

### `BriefingModal.tsx`
A detailed view of a selected prompt.
*   *Features:*
    *   Full syntax-highlighted content.
    *   "Copy Payload" button.
    *   Visual breakdown of the Alpha Score (charts/bars).

## 6. Core Loop (`hooks/usePromptFeed.ts`)
1.  **Poll:** `setInterval` triggers `fetch(feed.url)`.
2.  **Parse:** Adapts raw JSON (e.g., Reddit API) into `PromptObject`.
3.  **Filter:** Discards prompts below `UserPreferences.minAlphaThreshold`.
4.  **Score:** Runs `calculatePromptAlpha`.
5.  **Simulation Fallback:** If enabled, runs `simulationEngine.ts` to inject synthetic prompts.
6.  **Store:** Dispatches valid prompts to Zustand.

## 7. Performance Considerations
*   **Virtualization:** The main grid should eventually use virtualization if history exceeds 500 items.
*   **Storage:** `localStorage` is capped at ~5MB. The store uses `slice(0, maxItems)` to prevent overflow.
*   **Network:** Requests are direct from client. CORS issues may arise with some APIs; Reddit JSON is generally permissive for simple GETs.
