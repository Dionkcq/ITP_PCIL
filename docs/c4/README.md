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

## Interactive site (deployed)

The same model is published as an interactive, browsable site on GitHub Pages,
**auto-deployed** by [`.github/workflows/c4-pages.yml`](../../.github/workflows/c4-pages.yml)
on every push that touches `docs/c4/`:

**https://dionkcq.github.io/ITP_PCIL/**

Run it locally too:

```bash
npx -y likec4@latest start                          # dev server, live preview
npx -y likec4@latest build docs/c4 -o dist --base / # static site (dist/ gitignored)
```
