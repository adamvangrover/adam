1. Modify `services/webapp/client/src/hooks/useSwarmSimulation.ts` to extract only the `initializeSwarm` method, instead of `agents`, to prevent it from subscribing its parent (`PromptAlpha.tsx`) to every single state change from the simulation loop. It will use `.getState()` to access the latest state.
2. Replace `setInterval` in `services/webapp/client/src/hooks/useSwarmSimulation.ts` with a recursive `setTimeout` to follow the journal guidelines.
3. Test by running lint.
