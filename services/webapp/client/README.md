# ADAM Webapp Client

React 18 + Vite + Vitest. Node 22+.

| Command | What it does |
|---|---|
| `npm ci --legacy-peer-deps` | Install from `package-lock.json` |
| `npm run dev` | Dev server on http://localhost:3000; `/api` proxies to the Flask API on :5001 |
| `npm test` | Vitest (jsdom) — `src/**/*.test.{js,jsx,ts,tsx}` |
| `npm run build` | Production bundle in `build/` (served by `Dockerfile` / `nginx.conf`) |
| `npm run typecheck` | `tsc --noEmit` (not yet clean; not gated in CI) |

Files containing JSX use `.jsx`/`.tsx`; Vite does not parse JSX in `.js`.
