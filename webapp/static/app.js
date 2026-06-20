const SLOTS = 5;
const state = { device: "auto", source: "dataset", cards: [] };

const $ = (id) => document.getElementById(id);

function buildGrid() {
  const grid = $("grid");
  grid.innerHTML = "";
  state.cards = [];
  for (let i = 0; i < SLOTS; i++) {
    const card = document.createElement("div");
    card.className = "border border-outline-variant bg-[#0a0a0a] flex flex-col";
    card.innerHTML = `
      <div class="aspect-square relative">
        <div class="absolute inset-0 shimmer hidden" data-shimmer></div>
        <img class="w-full h-full object-cover hidden" data-img/>
        <div class="absolute inset-0 flex items-center justify-center tiny text-on-surface-variant opacity-40" data-empty>SLOT_${i}</div>
      </div>
      <div class="p-2 border-t border-outline-variant">
        <div class="tiny text-primary mb-2 truncate" data-name>VARIATION_${i}</div>
        <div class="grid grid-cols-2 gap-1 mb-2">
          <button class="py-1 border border-outline-variant tiny text-on-surface-variant" data-vote="rejected">REJECT</button>
          <button class="py-1 border border-outline-variant tiny text-on-surface-variant" data-vote="accepted">ACCEPT</button>
        </div>
        <input class="w-full bg-surface-container-low border border-outline-variant p-1 tiny text-on-surface" placeholder="comentario..." data-comment/>
      </div>`;
    grid.appendChild(card);
    const c = {
      el: card, vote: "pending",
      img: card.querySelector("[data-img]"),
      shimmer: card.querySelector("[data-shimmer]"),
      empty: card.querySelector("[data-empty]"),
      name: card.querySelector("[data-name]"),
      comment: card.querySelector("[data-comment]"),
    };
    card.querySelectorAll("[data-vote]").forEach((b) =>
      b.addEventListener("click", () => setVote(c, b.dataset.vote)));
    state.cards.push(c);
  }
}

function setVote(card, vote) {
  card.vote = vote;
  card.el.querySelectorAll("[data-vote]").forEach((b) => {
    const on = b.dataset.vote === vote;
    b.classList.toggle("bg-primary", on);
    b.classList.toggle("text-black", on);
    b.classList.toggle("text-on-surface-variant", !on);
  });
}

function setLoading(loading) {
  state.cards.forEach((c) => {
    c.shimmer.classList.toggle("hidden", !loading);
    if (loading) { c.empty.classList.add("hidden"); c.img.classList.add("hidden"); }
  });
  $("stream-indicator").hidden = !loading;
  $("node-status").textContent = loading ? "BUSY" : "IDLE";
  $("generate-btn").disabled = loading;
}

function fillCard(idx, name, dataUrl) {
  const c = state.cards[idx];
  if (!c) return;
  c.name.textContent = name;
  c.img.src = dataUrl;
  c.img.classList.remove("hidden");
  c.shimmer.classList.add("hidden");
  c.empty.classList.add("hidden");
}

async function streamNdjson(url, opts, onEvent) {
  const res = await fetch(url, opts);
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (line) onEvent(JSON.parse(line));
    }
  }
  if (buf.trim()) onEvent(JSON.parse(buf.trim()));
}

function fmt(secs) {
  if (secs < 60) return `~${Math.round(secs)}s`;
  return `~${Math.floor(secs / 60)}m ${Math.round(secs % 60)}s`;
}

async function generate() {
  buildGrid();
  setLoading(true);
  let done = 0;
  const deltas = [];
  let lastT = performance.now();
  $("progress-track").classList.remove("hidden");
  $("progress-bar").style.width = "0%";
  $("progress-label").textContent = `0 / ${SLOTS}`;
  $("status-text").textContent = "Inicializando pipeline (carga de modelos)...";

  const fd = new FormData();
  fd.append("device", state.device);
  fd.append("mode", state.source);
  if (state.source === "upload") {
    const file = $("upload-input").files[0];
    if (!file) { $("status-text").textContent = "Sube una imagen primero."; setLoading(false); return; }
    fd.append("image", file);
  } else {
    const scene = $("scene-select").value;
    if (!scene) { $("status-text").textContent = "No hay escenas locales. Descarga una del repositorio."; setLoading(false); return; }
    fd.append("scene_id", scene);
  }

  try {
    await streamNdjson("/api/generate", { method: "POST", body: fd }, (ev) => {
      if (ev.error) { $("status-text").textContent = "Error: " + ev.error; return; }
      if (ev.done) { $("status-text").textContent = "Listo. Evalua cada variacion."; return; }
      fillCard(ev.idx, ev.name, ev.image);
      done++;
      const now = performance.now();
      if (done > 1) deltas.push((now - lastT) / 1000);
      lastT = now;
      $("progress-bar").style.width = `${(done / SLOTS) * 100}%`;
      if (deltas.length) {
        const avg = deltas.reduce((a, b) => a + b, 0) / deltas.length;
        const eta = avg * (SLOTS - done);
        $("progress-label").textContent = `${done} / ${SLOTS} · ${fmt(eta)} restante`;
        $("status-text").textContent = `Generando... ${fmt(avg)} por imagen`;
      } else {
        $("progress-label").textContent = `${done} / ${SLOTS}`;
        $("status-text").textContent = `Generando... ${done}/${SLOTS}`;
      }
    });
  } catch (e) {
    $("status-text").textContent = "Error: " + e.message;
  } finally {
    setLoading(false);
    $("progress-track").classList.add("hidden");
  }
}

async function loadScenes() {
  const scenes = await (await fetch("/api/scenes")).json();
  const sel = $("scene-select");
  sel.innerHTML = scenes.length
    ? scenes.map((s) => `<option value="${s}">${s}</option>`).join("")
    : `<option value="">(sin escenas locales)</option>`;
  updatePreview();
}

function updatePreview() {
  const scene = $("scene-select").value;
  const img = $("scene-preview");
  if (scene) { img.src = `/api/scenes/${scene}/preview`; img.classList.remove("hidden"); }
  else { img.classList.add("hidden"); }
}

async function loadDatasets() {
  const list = $("dataset-list");
  try {
    const data = await (await fetch("/api/datasets")).json();
    if (data.error) { list.innerHTML = `<div class="tiny text-error p-2">${data.error}</div>`; return; }
    list.innerHTML = "";
    data.forEach((d) => {
      const row = document.createElement("div");
      row.className = "flex items-center justify-between gap-2 py-1";
      row.innerHTML = `<span class="tiny ${d.downloaded ? "text-primary" : "text-on-surface-variant"}">${d.id}</span>`;
      const btn = document.createElement("button");
      btn.className = "tiny px-2 py-1 border border-outline-variant text-on-surface-variant hover:bg-surface-container-highest";
      btn.textContent = d.downloaded ? "OK" : "GET";
      btn.disabled = d.downloaded;
      btn.addEventListener("click", () => downloadDataset(d.id, btn));
      row.appendChild(btn);
      list.appendChild(row);
    });
  } catch (e) {
    list.innerHTML = `<div class="tiny text-error p-2">${e.message}</div>`;
  }
}

async function downloadDataset(id, btn) {
  btn.disabled = true;
  const orig = btn.textContent;
  try {
    await streamNdjson(`/api/datasets/${id}/download`, { method: "POST" }, (ev) => {
      if (ev.error) { btn.textContent = "ERR"; $("status-text").textContent = ev.error; return; }
      if (ev.step) { btn.textContent = "..."; $("status-text").textContent = ev.step; }
      if (ev.done) { btn.textContent = "OK"; $("status-text").textContent = `${id} listo.`; }
    });
    await loadScenes();
  } catch (e) {
    btn.textContent = orig; btn.disabled = false;
    $("status-text").textContent = "Error: " + e.message;
  }
}

async function submitReport() {
  const out = $("report-output");
  out.classList.remove("hidden");

  const pending = state.cards.filter((c) => c.vote === "pending").length;
  if (pending) {
    out.textContent = `Faltan ${pending} variacion(es) por evaluar. Acepta o rechaza las 5 antes de generar el reporte.`;
    return;
  }

  const decisions = state.cards.map((c, i) => ({
    option: i + 1,
    style_name: c.name.textContent,
    status: c.vote,
    comment: c.comment.value,
  }));
  out.textContent = "Analizando decisiones...";
  try {
    const res = await fetch("/api/report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decisions }),
    });
    const data = await res.json();
    out.textContent = data.explanation || data.error || "Sin respuesta.";
  } catch (e) {
    out.textContent = "Error: " + e.message;
  }
}

function wireGroups() {
  $("device-group").querySelectorAll("button").forEach((b) =>
    b.addEventListener("click", () => {
      state.device = b.dataset.device;
      $("device-badge").textContent = b.dataset.device.toUpperCase();
      $("device-group").querySelectorAll("button").forEach((x) => {
        const on = x === b;
        x.classList.toggle("bg-primary", on);
        x.classList.toggle("text-black", on);
        x.classList.toggle("text-on-surface-variant", !on);
      });
    }));
  $("source-group").querySelectorAll("button").forEach((b) =>
    b.addEventListener("click", () => {
      state.source = b.dataset.source;
      $("dataset-panel").hidden = state.source !== "dataset";
      $("upload-panel").hidden = state.source !== "upload";
      $("source-group").querySelectorAll("button").forEach((x) => {
        const on = x === b;
        x.classList.toggle("bg-primary", on);
        x.classList.toggle("text-black", on);
        x.classList.toggle("text-on-surface-variant", !on);
      });
    }));
}

async function checkDevice() {
  try {
    const { gpu } = await (await fetch("/api/device")).json();
    if (!gpu) {
      const btn = $("device-group").querySelector('[data-device="gpu"]');
      btn.disabled = true;
      btn.classList.add("opacity-40", "cursor-not-allowed");
      btn.title = "Este servidor no tiene GPU";
    }
  } catch (e) {}
}

buildGrid();
wireGroups();
checkDevice();
loadScenes();
loadDatasets();
$("scene-select").addEventListener("change", updatePreview);
$("generate-btn").addEventListener("click", generate);
$("report-btn").addEventListener("click", submitReport);
