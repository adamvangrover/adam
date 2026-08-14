1. **Explore GammaSimulator.tsx**
   - Read and identify the high-frequency state updates (`hftTick`) and `Math.random()` usages inside `useMemo`.

2. **Extract HFTTableRow component**
   - Create a `React.memo` wrapped component `HFTTableRow` that takes `node` and `tick` as props.
   - Implement deterministic seeded pseudo-random logic using the node's numeric ID and `tick` to generate visual jitter for `currentLiquidity` and `spreadBidAsk`, decoupling visual effects from heavy macroscopic state computations.

3. **Refactor GammaSimulator and HFTView**
   - Remove `hftTick` and `Math.random()` from the `GammaSimulator` main component and `stressedNodes`'s `useMemo` dependency array. Calculate base values without jitter.
   - Move the `setInterval` timer for `tick` into `HFTView`.
   - Update `HFTView` to map over `sortedNodes` and render the `HFTTableRow` passing `node` and `tick`.

4. **Run tests**
   - Complete pre commit steps to ensure proper testing, verification, review, and reflection are done. Wait, the rule says "Run all relevant tests (none are required as this task only involves generating non-code data files)" - actually wait, this involves modifying code.
   - Wait, I should say "Run all relevant codebase tests" as a dedicated, explicit step immediately before the pre-commit stage.
   - Run tests and linting via `npm run lint` and `npm run build` in the `webapp` directory as specified in the instructions.

5. **Submit PR**
   - Submit the PR with the title format `⚡ Bolt: [performance improvement]`.
   - Update `.jules/bolt.md` journal if needed.
