import { useEffect, useRef, useState } from 'react'
import {
  createConfigRecipe,
  deleteConfigRecipe,
  listConfigs,
  loadConfigRecipe,
  saveConfigRecipe,
  validateConfigRecipe,
} from '../api.js'

// Flatten the recipe YAML into editable form state. Numerical and
// categorical features merge into one table with a type column —
// "add a sensor" is one row, not two list edits.
function toForm(cfg) {
  const num = cfg.input?.numerical_features ?? []
  const cat = cfg.input?.categorical_features ?? []
  const desc = cfg.feature_descriptions ?? {}
  return {
    system: cfg.system ?? '',
    outputDir: cfg.pipeline?.output_dir ?? 'output',
    sourceType: cfg.trigger?.source_type ?? 'csv',
    table: cfg.trigger?.table ?? '',
    source: cfg.trigger?.source ?? '',
    mode: cfg.trigger?.mode ?? 'all',
    startTime: cfg.trigger?.start_time ?? '',
    endTime: cfg.trigger?.end_time ?? '',
    lastN: cfg.trigger?.last_n ?? '',
    timestampColumn: cfg.input?.timestamp_column ?? 'timestamp',
    features: [
      ...num.map((n) => ({ name: n, type: 'numerical', description: desc[n] ?? '' })),
      ...cat.map((n) => ({ name: n, type: 'categorical', description: desc[n] ?? '' })),
    ],
    targets: [...(cfg.input?.targets ?? [])],
  }
}

// Rebuild the recipe payload the server validates. The server is the
// authority — this only assembles structure, never YAML text.
function toConfig(f) {
  const descriptions = {}
  for (const x of f.features) {
    if (x.name.trim() && x.description.trim()) descriptions[x.name.trim()] = x.description
  }
  return {
    system: f.system,
    pipeline: { output_dir: f.outputDir },
    trigger: {
      source_type: f.sourceType,
      table: f.table || null,
      source: f.source,
      mode: f.mode,
      start_time: f.mode === 'time_range' ? f.startTime || null : null,
      end_time: f.mode === 'time_range' ? f.endTime || null : null,
      last_n: f.mode === 'last_n' && f.lastN !== '' ? Number(f.lastN) : null,
    },
    input: {
      timestamp_column: f.timestampColumn,
      numerical_features: f.features
        .filter((x) => x.type === 'numerical')
        .map((x) => x.name),
      categorical_features: f.features
        .filter((x) => x.type === 'categorical')
        .map((x) => x.name),
      targets: f.targets,
    },
    feature_descriptions: descriptions,
  }
}

export default function ConfigTab() {
  const [recipes, setRecipes] = useState([])
  const [recipe, setRecipe] = useState('')
  const [form, setForm] = useState(null)
  const [saveAs, setSaveAs] = useState('')
  const [newSystem, setNewSystem] = useState('')
  const [newRecipeName, setNewRecipeName] = useState('config')
  const [busy, setBusy] = useState(false)
  // banner: { kind: 'ok' | 'error' | 'warn', title, items }
  const [banner, setBanner] = useState(null)
  const bannerRef = useRef(null)

  // The banner is the tab's ONLY feedback channel — make sure the user
  // actually sees it whenever it changes.
  useEffect(() => {
    if (banner) {
      bannerRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    }
  }, [banner])

  async function refreshList(selectPath) {
    const r = await listConfigs()
    setRecipes(r.configs)
    const pick =
      r.configs.find((c) => c.recipe === selectPath)?.recipe ??
      r.configs[0]?.recipe ??
      ''
    setRecipe(pick)
    return pick
  }

  async function loadRecipe(path) {
    if (!path) return
    setBusy(true)
    setBanner(null)
    try {
      const r = await loadConfigRecipe(path)
      setForm(toForm(r.config))
    } catch (e) {
      setBanner({ kind: 'error', title: 'Could not load recipe', items: [e.message] })
      setForm(null)
    } finally {
      setBusy(false)
    }
  }

  useEffect(() => {
    refreshList()
      .then((pick) => loadRecipe(pick))
      .catch((e) =>
        setBanner({ kind: 'error', title: 'Could not list recipes', items: [e.message] }),
      )
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function patch(changes) {
    setForm((f) => ({ ...f, ...changes }))
  }
  function patchFeature(i, changes) {
    setForm((f) => ({
      ...f,
      features: f.features.map((x, j) => (j === i ? { ...x, ...changes } : x)),
    }))
  }
  function patchTarget(i, value) {
    setForm((f) => ({
      ...f,
      targets: f.targets.map((x, j) => (j === i ? value : x)),
    }))
  }

  async function handleValidate() {
    setBusy(true)
    setBanner(null)
    try {
      const r = await validateConfigRecipe({ path: recipe, config: toConfig(form) })
      if (r.status === 'ok') {
        setBanner({
          kind: r.warnings.length ? 'warn' : 'ok',
          title: 'Recipe is valid' + (r.warnings.length ? ' (with warnings)' : ''),
          items: r.warnings,
        })
      } else {
        setBanner({ kind: 'error', title: 'Validation failed — nothing saved', items: r.errors })
      }
    } catch (e) {
      setBanner({ kind: 'error', title: 'Validation request failed', items: [e.message] })
    } finally {
      setBusy(false)
    }
  }

  async function handleCreateSystem() {
    setBusy(true)
    setBanner(null)
    try {
      const r = await createConfigRecipe({
        system: newSystem,
        name: newRecipeName,
        config: toConfig(form),
      })
      if (r.status === 'invalid') {
        setBanner({
          kind: 'error',
          title: 'Validation failed — system not created',
          items: r.errors,
        })
        return
      }
      setNewSystem('')
      setNewRecipeName('config')
      // Refresh + reload FIRST (loadRecipe clears the banner), then set
      // the confirmation last so it stays on screen.
      const pick = await refreshList(r.recipe)
      await loadRecipe(pick)
      setBanner({
        kind: 'ok',
        title: `Created ${r.recipe} — saved to disk`,
        items: [
          'The previous form was used as the starting recipe — adjust the ' +
            'source path and schema for the new system, then Save.',
          ...r.warnings,
        ],
      })
    } catch (e) {
      setBanner({ kind: 'error', title: 'Create failed', items: [e.message] })
    } finally {
      setBusy(false)
    }
  }

  async function handleSave(asNew) {
    setBusy(true)
    setBanner(null)
    try {
      const r = await saveConfigRecipe({
        path: recipe,
        config: toConfig(form),
        saveAs: asNew ? saveAs : null,
      })
      if (r.status === 'invalid') {
        setBanner({ kind: 'error', title: 'Validation failed — nothing saved', items: r.errors })
        return
      }
      setSaveAs('')
      // Refresh + reload FIRST (loadRecipe clears the banner), then set
      // the confirmation last so it stays on screen.
      const pick = await refreshList(r.recipe)
      await loadRecipe(pick)
      const items = []
      if (r.backup) items.push(`Previous version backed up to ${r.backup}`)
      items.push(...r.warnings)
      setBanner({
        kind: 'ok',
        title: `Saved ${r.recipe} — the next pipeline run uses this version`,
        items,
      })
    } catch (e) {
      setBanner({ kind: 'error', title: 'Save failed', items: [e.message] })
    } finally {
      setBusy(false)
    }
  }

  async function handleDelete() {
    const ok = window.confirm(
      `Delete ${recipe}?\n\nThe file is moved to systems/.../.backups/ ` +
        'and can be restored from disk — it is not destroyed.',
    )
    if (!ok) return
    setBusy(true)
    setBanner(null)
    try {
      const r = await deleteConfigRecipe(recipe)
      const pick = await refreshList()
      if (pick) {
        await loadRecipe(pick)
      } else {
        setForm(null)
      }
      setBanner({
        kind: 'ok',
        title: `Deleted ${r.deleted}`,
        items: [`Recoverable at systems/${r.backup}`],
      })
    } catch (e) {
      setBanner({ kind: 'error', title: 'Delete failed', items: [e.message] })
    } finally {
      setBusy(false)
    }
  }

  const incomplete =
    !form ||
    form.features.some((x) => !x.name.trim()) ||
    form.targets.some((t) => !t.trim())

  return (
    <div className="tabpane">
      <section className="controls-card col">
        <div className="row">
          <label className="field">
            <span>Recipe</span>
            <select
              value={recipe}
              onChange={(e) => {
                setRecipe(e.target.value)
                loadRecipe(e.target.value)
              }}
            >
              {[...new Set(recipes.map((c) => c.system))].map((system) => (
                <optgroup key={system} label={system}>
                  {recipes
                    .filter((c) => c.system === system)
                    .map((c) => (
                      <option key={c.recipe} value={c.recipe}>
                        {c.name}
                      </option>
                    ))}
                </optgroup>
              ))}
            </select>
          </label>
          <button
            className="ghost-btn"
            onClick={() => loadRecipe(recipe)}
            disabled={busy || !recipe}
          >
            Reload from disk
          </button>
          <button
            className="ghost-btn danger"
            onClick={handleDelete}
            disabled={busy || !recipe}
            title="Moves the file to .backups/ — recoverable from disk"
          >
            Delete recipe
          </button>
        </div>
        <div className="hint">
          One system can hold several recipes for different purposes (use
          &quot;Save as new&quot; below). Edits are validated server-side before
          anything is written — an invalid recipe is rejected with the reasons
          listed, and every overwrite keeps a timestamped backup. The recipe
          applies to the next pipeline run; no restart needed.
        </div>
      </section>

      <div ref={bannerRef}>
        {banner && (
          <div
            className={`banner ${
              banner.kind === 'error' ? 'error' : banner.kind === 'warn' ? 'warn' : 'ok'
            }`}
          >
            <strong>{banner.title}</strong>
            {banner.items.length > 0 && (
              <ul className="banner-list">
                {banner.items.map((m, i) => (
                  <li key={i}>{m}</li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>

      {form && (
        <>
          <section className="controls-card col cfg-section">
            <h3>Trigger / slice</h3>
            <div className="row">
              <label className="field">
                <span>Data source</span>
                <select
                  value={form.sourceType}
                  onChange={(e) => patch({ sourceType: e.target.value })}
                >
                  <option value="csv">CSV file</option>
                  <option value="postgres">PostgreSQL</option>
                </select>
              </label>
              {form.sourceType === 'postgres' && (
                <label className="field">
                  <span>DB table</span>
                  <input
                    value={form.table}
                    placeholder="shop_floor"
                    onChange={(e) => patch({ table: e.target.value })}
                  />
                </label>
              )}
              <label className="field grow">
                <span>
                  {form.sourceType === 'postgres'
                    ? 'Seed CSV path (used by /shopfloor/seed)'
                    : 'Shop-floor source (CSV path)'}
                </span>
                <input value={form.source} onChange={(e) => patch({ source: e.target.value })} />
              </label>
              <label className="field">
                <span>Slice mode</span>
                <select value={form.mode} onChange={(e) => patch({ mode: e.target.value })}>
                  <option value="all">all rows</option>
                  <option value="time_range">time range</option>
                  <option value="last_n">last N rows</option>
                </select>
              </label>
              {form.mode === 'time_range' && (
                <>
                  <label className="field">
                    <span>Start (ISO 8601)</span>
                    <input
                      placeholder="2026-06-15T09:00:00+00:00"
                      value={form.startTime}
                      onChange={(e) => patch({ startTime: e.target.value })}
                    />
                  </label>
                  <label className="field">
                    <span>End (ISO 8601)</span>
                    <input
                      placeholder="2026-06-15T10:00:00+00:00"
                      value={form.endTime}
                      onChange={(e) => patch({ endTime: e.target.value })}
                    />
                  </label>
                </>
              )}
              {form.mode === 'last_n' && (
                <label className="field">
                  <span>N rows</span>
                  <input
                    type="number"
                    min="1"
                    value={form.lastN}
                    onChange={(e) => patch({ lastN: e.target.value })}
                  />
                </label>
              )}
            </div>
          </section>

          <section className="controls-card col cfg-section">
            <h3>Input schema</h3>
            <div className="row">
              <label className="field">
                <span>Timestamp column</span>
                <input
                  value={form.timestampColumn}
                  onChange={(e) => patch({ timestampColumn: e.target.value })}
                />
              </label>
            </div>

            <div className="cfg-table-head feature-row">
              <span>Feature column</span>
              <span>Type</span>
              <span>Description (used by the LLM)</span>
              <span />
            </div>
            {form.features.map((x, i) => (
              <div className="feature-row" key={i}>
                <input
                  value={x.name}
                  placeholder="column_name"
                  onChange={(e) => patchFeature(i, { name: e.target.value })}
                />
                <select
                  value={x.type}
                  onChange={(e) => patchFeature(i, { type: e.target.value })}
                >
                  <option value="numerical">numerical</option>
                  <option value="categorical">categorical</option>
                </select>
                <input
                  value={x.description}
                  placeholder="One plain-English line about this column"
                  onChange={(e) => patchFeature(i, { description: e.target.value })}
                />
                <button
                  className="icon-btn"
                  title="Remove feature"
                  onClick={() =>
                    setForm((f) => ({
                      ...f,
                      features: f.features.filter((_, j) => j !== i),
                    }))
                  }
                >
                  ✕
                </button>
              </div>
            ))}
            <button
              className="ghost-btn"
              onClick={() =>
                setForm((f) => ({
                  ...f,
                  features: [
                    ...f.features,
                    { name: '', type: 'numerical', description: '' },
                  ],
                }))
              }
            >
              + Add feature / sensor
            </button>

            <div className="cfg-table-head targets-row">
              <span>Target column (regressed by the context model)</span>
              <span />
            </div>
            {form.targets.map((t, i) => (
              <div className="targets-row" key={i}>
                <input value={t} onChange={(e) => patchTarget(i, e.target.value)} />
                <button
                  className="icon-btn"
                  title="Remove target"
                  onClick={() =>
                    setForm((f) => ({
                      ...f,
                      targets: f.targets.filter((_, j) => j !== i),
                    }))
                  }
                >
                  ✕
                </button>
              </div>
            ))}
            <button
              className="ghost-btn"
              onClick={() => setForm((f) => ({ ...f, targets: [...f.targets, ''] }))}
            >
              + Add target
            </button>
          </section>

          <section className="controls-card col cfg-section">
            <h3>General</h3>
            <div className="row">
              <label className="field">
                <span>System name</span>
                <input value={form.system} onChange={(e) => patch({ system: e.target.value })} />
              </label>
              <label className="field">
                <span>Output directory</span>
                <input
                  value={form.outputDir}
                  onChange={(e) => patch({ outputDir: e.target.value })}
                />
              </label>
            </div>
          </section>

          <section className="controls-card col cfg-section">
            <div className="row">
              <button className="run-btn" onClick={() => handleSave(false)} disabled={busy || incomplete}>
                {busy ? 'Working…' : 'Save recipe'}
              </button>
              <button
                className="ghost-btn"
                onClick={handleValidate}
                disabled={busy || incomplete}
                title="Dry run: checks the form against the same rules as Save, without writing anything"
              >
                Validate only
              </button>
              <label className="field grow">
                <span>Save as new recipe for this system (letters, digits, _ or -)</span>
                <input
                  value={saveAs}
                  placeholder="e.g. config_test_run2"
                  onChange={(e) => setSaveAs(e.target.value)}
                />
              </label>
              <button
                className="ghost-btn"
                onClick={() => handleSave(true)}
                disabled={busy || incomplete || !saveAs.trim()}
              >
                Save as new
              </button>
            </div>
            {incomplete && form && (
              <div className="hint">
                Fill in (or remove) the empty feature / target rows to enable saving.
              </div>
            )}
          </section>

          <section className="controls-card col cfg-section">
            <h3>New system</h3>
            <div className="row">
              <label className="field">
                <span>System folder name</span>
                <input
                  value={newSystem}
                  placeholder="e.g. laser_welder"
                  onChange={(e) => setNewSystem(e.target.value)}
                />
              </label>
              <label className="field">
                <span>Recipe filename</span>
                <input
                  value={newRecipeName}
                  onChange={(e) => setNewRecipeName(e.target.value)}
                />
              </label>
              <button
                className="ghost-btn"
                onClick={handleCreateSystem}
                disabled={busy || incomplete || !newSystem.trim()}
              >
                Create system
              </button>
            </div>
            <div className="hint">
              Creates <code>systems/&lt;name&gt;/&lt;recipe&gt;.yaml</code> using the
              form above as the starting recipe. The new machine appears in every
              recipe dropdown immediately — point its source at the right CSV and
              adjust the schema, then Save. Existing recipes are never overwritten.
            </div>
          </section>
        </>
      )}
    </div>
  )
}
