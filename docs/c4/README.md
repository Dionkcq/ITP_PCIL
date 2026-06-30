# PCIL C4 diagrams (LikeC4)

The architecture diagrams are generated from LikeC4 models. There are **two
models, one per branch**, kept in separate folders so their element ids never
clash (LikeC4 scans a build folder recursively):

| Folder | Model | Describes |
|---|---|---|
| [`main/`](main/) | [`main/pcil.c4`](main/pcil.c4) | the `main` branch: CSV/file source, single orchestrator container. The PNGs the project [README](../../README.md#architecture-c4) embeds live here. |
| [`postgres/`](postgres/) | [`postgres/pcil.c4`](postgres/pcil.c4) | the `postgre_implementation` branch: PostgreSQL (pgvector) source, hybrid RAG, pipeline/anomaly container split. |
| [`landing/`](landing/) | `landing/index.html` | the chooser page served at the Pages root, linking both. |

One model, multiple consistent views (define an element once; LikeC4 rolls
relationships up per view).

## Edit

Edit the relevant `pcil.c4`. For live preview while editing, install the
**LikeC4** VS Code extension (it renders the views as you type).

> The `postgres/pcil.c4` here is a published-docs copy of the model on the
> `postgre_implementation` branch. When that branch merges to `main`, collapse
> back to a single model.

## Regenerate the PNGs (main only — these are what the README embeds)

From this folder (`docs/c4/`):

```bash
npx -y likec4@latest export png -o main main
```

This rewrites the five PNGs the README embeds:

| File | View |
|---|---|
| `main/index.png` | Level 1 - System Context |
| `main/containers.png` | Level 2 - Containers |
| `main/components.png` | Level 3 - Components |
| `main/dataflow.png` | Runtime data-flow (diagnosis) |
| `main/anomalyflow.png` | Anomaly scoring flow |

The first run downloads a headless browser via `npx`; no dependency is added to
the project (it is **not** in the dashboard's `package.json`, so the Docker build
is unaffected).

## Interactive site (deployed)

Both models are published as interactive sites on GitHub Pages, **auto-deployed**
by [`.github/workflows/c4-pages.yml`](../../.github/workflows/c4-pages.yml) on
every push that touches `docs/c4/`. ONE Pages site, two sub-path pages:

| URL | Page |
|---|---|
| **https://dionkcq.github.io/ITP_PCIL/** | landing chooser |
| **https://dionkcq.github.io/ITP_PCIL/main/** | main (CSV) architecture |
| **https://dionkcq.github.io/ITP_PCIL/postgres/** | Postgres-branch architecture |

Run them locally too:

```bash
npx -y likec4@latest start docs/c4/main                       # dev server, live preview
npx -y likec4@latest build docs/c4/main -o dist --base /      # static site (dist/ gitignored)
```
