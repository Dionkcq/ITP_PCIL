const runBtn = document.getElementById("runBtn");
const statusEl = document.getElementById("status");
const recommendationEl = document.getElementById("recommendation");
const contextEl = document.getElementById("contextWindow");
const recordsEl = document.getElementById("records");
const recordCountEl = document.getElementById("recordCount");
const rawJsonEl = document.getElementById("rawJson");
const toggleRawBtn = document.getElementById("toggleRaw");

let rawVisible = false;

const setStatus = (text, tone = "muted") => {
  statusEl.textContent = text;
  statusEl.className = `status ${tone}`;
};

const renderContext = (cw) => {
  if (!cw) {
    contextEl.textContent = "No context window available.";
    return;
  }
  contextEl.innerHTML = `
    <div>Start: ${cw.start_time}</div>
    <div>End: ${cw.end_time}</div>
    <div>Rows: ${cw.row_count}</div>
    <div>Features: ${cw.feature_count}</div>
    <div>Targets: ${cw.target_count}</div>
  `;
};

const renderRecords = (records) => {
  if (!records || records.length === 0) {
    recordsEl.textContent = "No matching recovery records.";
    recordCountEl.textContent = "0";
    return;
  }

  recordCountEl.textContent = String(records.length);
  recordsEl.innerHTML = records
    .map(
      (rec, index) => `
      <div class="record">
        <h3>Record ${index + 1} - ${rec.source_doc}</h3>
        <p><strong>Error:</strong> ${rec.error}</p>
        <p><strong>Cause:</strong> ${rec.cause}</p>
        <p><strong>Recovery:</strong> ${rec.recovery}</p>
      </div>
    `,
    )
    .join("");
};

const renderRaw = (data) => {
  rawJsonEl.textContent = JSON.stringify(data, null, 2);
};

const runPipeline = async () => {
  runBtn.disabled = true;
  setStatus("Running pipeline...", "busy");
  recommendationEl.textContent = "";
  contextEl.textContent = "";
  recordsEl.textContent = "";

  try {
    const resp = await fetch("/run", { method: "POST" });
    const payload = await resp.json();

    if (!resp.ok) {
      throw new Error(payload.error || "Pipeline failed.");
    }

    recommendationEl.textContent = payload.operator_recommendation || "No output.";
    renderContext(payload.impacts?.context_window);
    renderRecords(payload.recovery_records || []);
    renderRaw(payload);
    setStatus("Run complete", "ok");
  } catch (err) {
    recommendationEl.textContent = "";
    contextEl.textContent = "";
    recordsEl.textContent = "";
    renderRaw({ error: err.message });
    setStatus(`Error: ${err.message}`, "error");
  } finally {
    runBtn.disabled = false;
  }
};

runBtn.addEventListener("click", runPipeline);

toggleRawBtn.addEventListener("click", () => {
  rawVisible = !rawVisible;
  rawJsonEl.classList.toggle("hidden", !rawVisible);
  toggleRawBtn.textContent = rawVisible ? "Hide JSON" : "Show JSON";
});
