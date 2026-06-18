# PCIL C4 diagrams (LikeC4)

The architecture diagrams in the project [README](../../README.md#architecture-c4)
are generated from a single LikeC4 model: [`pcil.c4`](pcil.c4). One model, multiple
consistent views (define an element once; LikeC4 rolls relationships up per view).

## Edit

Edit `pcil.c4`. For live preview while editing, install the **LikeC4** VS Code
extension (it renders the views as you type).

## Regenerate the PNGs

From this folder (`docs/c4/`):

```bash
npx -y likec4@latest export png -o .
```

This rewrites the five PNGs the README embeds:

| File | View |
|---|---|
| `index.png` | Level 1 - System Context |
| `containers.png` | Level 2 - Containers |
| `components.png` | Level 3 - Components |
| `dataflow.png` | Runtime data-flow (diagnosis) |
| `anomalyflow.png` | Anomaly scoring flow |

The first run downloads a headless browser via `npx`; no dependency is added to the
project (it is **not** in the dashboard's `package.json`, so the Docker build is
unaffected).

## Optional: interactive site

LikeC4 can also build an interactive, browsable version of the same model:

```bash
npx -y likec4@latest build -o dist   # static site (dist/ is gitignored)
npx -y likec4@latest start           # local dev server with live preview
```
