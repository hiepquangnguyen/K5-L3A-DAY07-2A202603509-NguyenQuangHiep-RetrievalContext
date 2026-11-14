"use strict";

const state = {
  overview: null,
  activeTab: "live",
  kb: "advanced",
  mode: "advanced",
  isAsking: false,
  liveResult: null,
  livePage: 1,
  liveRegion: "cached",
  basicRetSplit: "dev",
  basicAgentSplit: "dev",
  basicCaseId: null,
  advSplit: "test",
  advFilter: "all",
  advCaseId: null,
  corpusKb: "advanced",
  corpusPage: 1,
  corpusModality: "all",
  corpusSource: "",
  corpusQuery: "",
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function esc(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function fmtUsd(val) {
  const num = Number(val || 0);
  return `$${num.toFixed(6).replace(/0+$/, "").replace(/\.$/, ".00")}`;
}

function fmtPct(rate) {
  return `${(Number(rate || 0) * 100).toFixed(1).replace(/\.0$/, "")}%`;
}

function statusPill(st) {
  if (st === "answered") return `<span class="pill pill-ok">ANSWERED</span>`;
  if (st === "abstain") return `<span class="pill pill-warn">ABSTAIN</span>`;
  return `<span class="pill pill-err">${esc(st || "ERROR")}</span>`;
}

function renderAnswerWithCitations(answerText, validCitations, chunkDetails) {
  const validSet = new Set(validCitations || []);
  const parts = String(answerText || "").split(/(\[[a-z0-9:_-]+\])/gi);
  return parts
    .map((part) => {
      const m = /^\[([a-z0-9:_-]+)\]$/i.exec(part);
      if (!m) return esc(part);
      const cid = m[1];
      const known = validSet.has(cid) || Boolean((chunkDetails || {})[cid]);
      const cls = known ? "cit-pill" : "cit-pill orphan";
      return `<button type="button" class="${cls}" data-cit-id="${esc(cid)}" title="Xem chi tiết đoạn trích ${esc(cid)}">[${esc(cid)}]</button>`;
    })
    .join("");
}

function renderTraceEvents(events) {
  const list = events || [];
  if (!list.length) return `<p class="meta-hint">Không có bước thực thi được ghi lại.</p>`;
  return list
    .map((ev, idx) => {
      const evType = ev.event || "step";
      const toolName = ev.tool || evType;
      if (evType === "answer" || evType === "abstain") {
        return `
          <div class="trace-step">
            <div class="trace-step-head">
              <span class="trace-tool-name">#${idx + 1} · ${esc(toolName)}</span>
              ${statusPill(evType === "answer" ? "answered" : "abstain")}
            </div>
          </div>`;
      }
      if (evType === "invalid_or_missing_citations" || evType === "search_limit" || evType === "turn_limit") {
        return `
          <div class="trace-step" style="border-color:#fde68a;background:#fffbeb;">
            <div class="trace-step-head">
              <span class="trace-tool-name" style="color:#b45309;">#${idx + 1} · ${esc(evType)}</span>
              <span class="pill pill-warn">Cơ chế tự kiểm soát</span>
            </div>
          </div>`;
      }
      const hits = (ev.results || [])
        .map((r) => {
          const cid = typeof r === "string" ? r : r.id;
          const pg = typeof r === "object" && r.page ? ` p.${r.page}` : "";
          const mod = typeof r === "object" && r.modality ? ` (${r.modality})` : "";
          return `<button type="button" class="hit-chip" data-cit-id="${esc(cid)}">${esc(cid)}${esc(pg)}${esc(mod)}</button>`;
        })
        .join("");
      const visHtml = ev.visual_observation
        ? `<div class="trace-vision"><strong>Vision Observation:</strong>\n${esc(ev.visual_observation)}</div>`
        : "";
      return `
        <div class="trace-step">
          <div class="trace-step-head">
            <span class="trace-tool-name">#${idx + 1} · ${esc(toolName)}</span>
            <span class="pill pill-indigo">${(ev.results || []).length} kết quả</span>
          </div>
          ${ev.query ? `<div class="trace-query"><strong>Truy vấn/Tham số:</strong> <code>${esc(ev.query)}</code></div>` : ""}
          ${hits ? `<div class="hit-chips">${hits}</div>` : ""}
          ${visHtml}
        </div>`;
    })
    .join("");
}

function openLightbox(src, caption) {
  $("#lightbox-img").src = src;
  $("#lightbox-caption").textContent = caption;
  $("#lightbox").classList.remove("hidden");
}

function closeLightbox() {
  $("#lightbox").classList.add("hidden");
  $("#lightbox-img").removeAttribute("src");
}

function switchTab(tabName) {
  state.activeTab = tabName;
  $$(".nav-btn").forEach((btn) => btn.classList.toggle("active", btn.dataset.tab === tabName));
  $$(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === `tab-${tabName}`));
}

function syncModeUI() {
  $$(".kb-switch .seg-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.kb === state.kb && b.dataset.mode === state.mode);
  });
  $("#adv-options").classList.toggle("hidden", state.kb !== "advanced");
  $("#basic-options").classList.toggle("hidden", state.kb !== "basic");
}

function renderStatusStrip(status, overview) {
  const pills = [];
  pills.push(
    status.openai_configured
      ? `<span class="pill pill-ok">OpenAI Live (${esc(status.chat_model)})</span>`
      : `<span class="pill pill-err">Thiếu OPENAI_API_KEY</span>`
  );
  pills.push(
    status.basic_qdrant_ready
      ? `<span class="pill pill-indigo">MOET Qdrant · ${overview.basic.chunk_count} chunks</span>`
      : `<span class="pill pill-warn">MOET Qdrant chưa sẵn sàng</span>`
  );
  pills.push(
    status.adv_qdrant_ready
      ? `<span class="pill pill-indigo">FPT BCTN 2025 · ${overview.advanced.chunk_count} chunks · ${status.adv_pages_count} PNG</span>`
      : `<span class="pill pill-warn">FPT Qdrant chưa sẵn sàng</span>`
  );
  $("#status-strip").innerHTML = pills.join("");
}

function renderPresets(presets) {
  $("#preset-chips").innerHTML = (presets || [])
    .map(
      (p) =>
        `<button type="button" class="preset-chip" data-preset-id="${esc(p.id)}" title="${esc(p.question)}">${esc(p.tag)}</button>`
    )
    .join("");
}

function renderContextList(trace, highlightId = null) {
  const details = trace.chunk_details || {};
  const ctxIds = trace.context_ids && trace.context_ids.length ? trace.context_ids : Object.keys(details);
  const citedSet = new Set(trace.citations || []);
  $("#ctx-count").textContent = String(ctxIds.length);
  if (!ctxIds.length) {
    $("#live-context-list").innerHTML = `<p class="meta-hint">Không có chunk nào trong packed context.</p>`;
    return;
  }
  $("#live-context-list").innerHTML = ctxIds
    .map((cid) => {
      const ch = details[cid] || { id: cid, page: 0, modality: "text", section: "", text: "(Không tìm thấy nội dung chunk)" };
      const isCited = citedSet.has(cid);
      const isHl = highlightId === cid;
      const modPill =
        ch.modality === "multimodal"
          ? `<span class="pill pill-sky">VISION :v</span>`
          : `<span class="pill pill-muted">TEXT</span>`;
      const citeTag = isCited ? `<span class="pill pill-ok">Đã trích dẫn</span>` : "";
      const jumpBtn =
        trace.telemetry?.kb === "advanced" && ch.page
          ? `<button type="button" class="btn-secondary" data-jump-page="${ch.page}">Xem tr.${ch.page}</button>`
          : "";
      return `
        <div class="chunk-card ${isCited ? "is-cited" : ""} ${isHl ? "is-highlighted" : ""}" id="chunk-card-${esc(cid)}">
          <div class="chunk-card-head">
            <span class="chunk-id">${esc(cid)}</span>
            <div style="display:flex;gap:5px;align-items:center;">
              ${modPill}
              ${ch.page ? `<span class="pill pill-muted">Tr. ${ch.page}</span>` : ""}
              ${citeTag}
              ${jumpBtn}
            </div>
          </div>
          ${ch.section ? `<div class="chunk-section">${esc(ch.section)}</div>` : ""}
          <div class="chunk-text">${esc(ch.text)}</div>
        </div>`;
    })
    .join("");
}

function renderLivePageViewer(kb, availablePages) {
  const viewer = $("#live-page-viewer");
  if (kb !== "advanced") {
    viewer.innerHTML = `
      <div style="padding:18px;text-align:center;color:#475569;">
        <p><strong>Bộ dữ liệu MOET Quy chế (Day 6)</strong></p>
        <p style="font-size:12.5px;">Các trích dẫn ở chế độ MOET trỏ trực tiếp tới 210 chunk văn bản trong Qdrant Day 6. Chuyển sang tab <strong>Context đóng gói</strong> bên cạnh hoặc nhấn vào từng thẻ <code>[chunk-id]</code> để xem nguyên văn.</p>
      </div>`;
    return;
  }
  const pg = state.livePage || (availablePages && availablePages[0]) || 1;
  const src = `/api/advanced/page/${pg}?region=${encodeURIComponent(state.liveRegion)}`;
  const pageBtns = (availablePages && availablePages.length ? availablePages : [pg])
    .map(
      (p) =>
        `<button type="button" class="page-pill-btn ${p === pg ? "active" : ""}" data-jump-page="${p}">Tr.${p}</button>`
    )
    .join("");
  viewer.innerHTML = `
    <div class="page-toolbar">
      <div class="page-pills">${pageBtns}</div>
      <div style="display:flex;gap:6px;align-items:center;">
        <select id="live-region-sel" class="btn-secondary" aria-label="Vùng zoom PDF">
          <option value="cached" ${state.liveRegion === "cached" ? "selected" : ""}>Cached Full (150 DPI)</option>
          <option value="full" ${state.liveRegion === "full" ? "selected" : ""}>Live Full (200 DPI)</option>
          <option value="top" ${state.liveRegion === "top" ? "selected" : ""}>Zoom Nửa Trên (240 DPI)</option>
          <option value="bottom" ${state.liveRegion === "bottom" ? "selected" : ""}>Zoom Nửa Dưới (240 DPI)</option>
          <option value="left" ${state.liveRegion === "left" ? "selected" : ""}>Zoom Nửa Trái (240 DPI)</option>
          <option value="right" ${state.liveRegion === "right" ? "selected" : ""}>Zoom Nửa Phải (240 DPI)</option>
        </select>
        <button type="button" class="btn-secondary" id="live-zoom-modal-btn">Phóng to</button>
      </div>
    </div>
    <div class="page-frame">
      <img id="live-page-img" src="${src}" alt="FPT BCTN 2025 Trang ${pg}" />
    </div>
    <div class="meta-hint" style="margin-top:6px;">Nhấp vào ảnh để mở toàn màn hình · Trang ${pg}/232 (${esc(state.liveRegion)})</div>
  `;
}

function renderAnswerCardHtml(trace, titlePrefix = "") {
  const tel = trace.telemetry || {};
  const tokTotal = (tel.agent_llm_tokens || trace.llm_tokens || 0) + (tel.embed_tokens || 0) + (tel.rerank_tokens || 0);
  const costTotal = tel.total_cost_usd ?? trace.llm_cost_usd ?? 0;
  const steps = (trace.events || []).filter((e) => e.event === "search").length;
  return `
    <div class="answer-header">
      <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
        ${titlePrefix ? `<strong>${esc(titlePrefix)}</strong>` : ""}
        ${statusPill(trace.status)}
        <span class="pill pill-indigo">${(trace.citations || []).length} citations hợp lệ</span>
      </div>
      <span class="meta-hint">${esc(tel.chat_model || "gpt-4.1-mini")} · ${(trace.elapsed_ms || 0).toFixed(0)} ms</span>
    </div>
    <div class="answer-body">${renderAnswerWithCitations(trace.answer, trace.citations, trace.chunk_details)}</div>
    <div class="telemetry-bar">
      <span><strong>Latency:</strong> ${Number(trace.elapsed_ms || 0).toFixed(1)} ms</span>
      <span><strong>Tool calls:</strong> ${steps}</span>
      <span><strong>Agent tokens:</strong> ${Number(tel.agent_llm_tokens ?? trace.llm_tokens ?? 0).toLocaleString()}</span>
      <span><strong>Embed tokens:</strong> ${Number(tel.embed_tokens || 0).toLocaleString()}</span>
      <span><strong>Rerank tokens:</strong> ${Number(tel.rerank_tokens || 0).toLocaleString()}</span>
      <span><strong>Tổng tokens:</strong> ${Number(tokTotal).toLocaleString()}</span>
      <span><strong>Chi phí ước tính:</strong> ${fmtUsd(costTotal)}</span>
    </div>
  `;
}

function renderLiveResult(data) {
  state.liveResult = data;
  $("#live-empty").classList.add("hidden");
  $("#live-error").classList.add("hidden");

  if (data.comparison) {
    $("#live-workspace").classList.add("hidden");
    const wrap = $("#live-compare-wrap");
    wrap.classList.remove("hidden");
    wrap.innerHTML = ["baseline", "chosen"]
      .map((m) => {
        const tr = data.comparison[m];
        const heading =
          m === "baseline"
            ? "Baseline Starter (dense_qdrant_800 · 1 tool)"
            : "Chosen Multi-Tool (hybrid_rerank_pack2400 · 3 tools)";
        const borderStyle = m === "chosen" ? "border-top:4px solid #047857;" : "border-top:4px solid #b45309;";
        const ctxBlocks = (tr.context_ids || []).length
          ? tr.context_ids
              .map((cid) => {
                const ch = (tr.chunk_details || {})[cid] || { text: "", page: "" };
                return `<div class="chunk-card ${(tr.citations || []).includes(cid) ? "is-cited" : ""}">
                  <div class="chunk-card-head"><span class="chunk-id">${esc(cid)}</span><span class="pill pill-muted">Tr.${esc(ch.page)}</span></div>
                  <div class="chunk-text">${esc(ch.text)}</div>
                </div>`;
              })
              .join("")
          : `<p class="meta-hint">Context rỗng hoặc không nạp được đoạn chứa bằng chứng.</p>`;
        return `
          <div class="card" style="${borderStyle}">
            ${renderAnswerCardHtml(tr, heading)}
            <h3 class="sub-heading">Trace thực thi</h3>
            <div class="trace-list">${renderTraceEvents(tr.events)}</div>
            <h3 class="sub-heading">Packed Context (${(tr.context_ids || []).length})</h3>
            <div class="chunk-stack" style="max-height:300px;">${ctxBlocks}</div>
          </div>`;
      })
      .join("");
    return;
  }

  $("#live-compare-wrap").classList.add("hidden");
  $("#live-workspace").classList.remove("hidden");
  const primary = data.primary;
  $("#live-answer-card").innerHTML = renderAnswerCardHtml(
    primary,
    data.kb === "advanced" ? "Advanced Multimodal Agent" : "MOET Chosen Agent"
  );
  $("#live-trace-card").innerHTML = `
    <h3 class="trace-title">Trace điều phối công cụ &amp; Vision (${(primary.events || []).length} sự kiện)</h3>
    <div class="trace-list">${renderTraceEvents(primary.events)}</div>
  `;

  state.liveRegion = "cached";
  state.livePage = (primary.pages && primary.pages[0]) || 1;
  renderLivePageViewer(data.kb, primary.pages || []);
  renderContextList(primary);

  if (data.kb !== "advanced") {
    activateSideTab("context");
  } else {
    activateSideTab("page");
  }
}

function activateSideTab(name) {
  $$(".side-tab-btn").forEach((b) => b.classList.toggle("active", b.dataset.sideTab === name));
  $$(".side-pane").forEach((p) => p.classList.toggle("active", p.id === `side-pane-${name}`));
}

function jumpToCitation(cid) {
  if (!state.liveResult || state.liveResult.comparison) return;
  const primary = state.liveResult.primary;
  const ch = (primary.chunk_details || {})[cid];
  if (!ch) return;
  if (state.liveResult.kb === "advanced" && ch.page) {
    state.livePage = ch.page;
    renderLivePageViewer("advanced", primary.pages || [ch.page]);
  }
  renderContextList(primary, cid);
  activateSideTab("context");
  const el = document.getElementById(`chunk-card-${cid}`);
  if (el) el.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function askLive(questionText) {
  const q = questionText.trim();
  if (!q || state.isAsking) return;
  state.isAsking = true;
  $("#ask-submit").disabled = true;
  $("#live-error").classList.add("hidden");
  $("#live-loading").classList.remove("hidden");

  try {
    const resp = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: q,
        kb: state.kb,
        mode: state.mode,
        filter_cap: $("#opt-filter-cap").value || null,
        use_reranker: $("#opt-reranker").checked,
      }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
    renderLiveResult(data);
  } catch (err) {
    $("#live-error").textContent = `Lỗi thực thi: ${err.message}`;
    $("#live-error").classList.remove("hidden");
  } finally {
    state.isAsking = false;
    $("#ask-submit").disabled = false;
    $("#live-loading").classList.add("hidden");
  }
}

/* ==================== TAB 2: BASIC BENCHMARK ==================== */
function renderBasicTab() {
  const b = state.overview.basic;
  const devBase = b.agent.dev.baseline;
  const devChosen = b.agent.dev.chosen;
  const testChosen = b.agent.test.chosen;
  $("#basic-kpi-strip").innerHTML = `
    <div class="kpi-card">
      <div class="kpi-label">Kho dữ liệu Day 6 (Chiến lược C)</div>
      <div class="kpi-val">${b.chunk_count} chunks</div>
      <div class="kpi-sub">tt08-2021 (${b.sources["tt08-2021"] || 0}) · tt18-2021 (${b.sources["tt18-2021"] || 0})</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Retrieval Chosen (hybrid_rerank_pack2400)</div>
      <div class="kpi-val" style="color:var(--emerald);">20/20 dev · 6/6 test</div>
      <div class="kpi-sub">Hit@1 = 100%, Hit@3 Strict = 100%, Context Hit = 100%</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Agent Grounded Mean (Baseline → Chosen)</div>
      <div class="kpi-val">${devBase.summary.grounded_answer?.mean?.toFixed(2)} → <span style="color:var(--emerald);">${devChosen.summary.grounded_answer?.mean?.toFixed(2)}</span></div>
      <div class="kpi-sub">test Chosen: ${testChosen.summary.grounded_answer?.N}/${testChosen.summary.grounded_answer?.N} điểm 2/2</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">False Abstention (Từ chối sai)</div>
      <div class="kpi-val">${devBase.summary.false_abstention?.n}/${devBase.summary.false_abstention?.N} → <span style="color:var(--emerald);">${devChosen.summary.false_abstention?.n}/${devChosen.summary.false_abstention?.N}</span></div>
      <div class="kpi-sub">test: ${b.agent.test.baseline.summary.false_abstention?.n}/${b.agent.test.baseline.summary.false_abstention?.N} → ${b.agent.test.chosen.summary.false_abstention?.n}/${b.agent.test.chosen.summary.false_abstention?.N} · abstention đúng ${testChosen.summary.abstention?.n}/${testChosen.summary.abstention?.N}</div>
    </div>
  `;
  renderBasicRetrieval();
  renderBasicAgent();
}

function renderBasicRetrieval() {
  const split = state.basicRetSplit;
  $$("#basic-ret-split .seg-btn").forEach((b) => b.classList.toggle("active", b.dataset.split === split));
  const rows = state.overview.basic.retrieval[split] || [];
  $("#basic-ret-bars").innerHTML = rows
    .map((r) => {
      const isBest = r.config === "hybrid_rerank_pack2400";
      return `
        <div class="bar-row">
          <div class="bar-label">
            <span>${esc(r.config)}</span>
            <span>Hit@1: ${r.hit1}/${r.n} (${fmtPct(r.hit1_rate)})</span>
          </div>
          <div class="bar-track"><div class="bar-fill ${isBest ? "best" : ""}" style="width:${Number(r.hit1_rate || 0) * 100}%"></div></div>
          <div class="bar-sub">Context Hit: ${r.context_hit}/${r.n} (${fmtPct(r.context_hit_rate)}) · Strict Hit@3: ${r.hit3_strict}/${r.n} · ${r.median_ms} ms</div>
        </div>`;
    })
    .join("");

  $("#basic-ret-table").innerHTML = `
    <table class="data-table">
      <thead>
        <tr>
          <th>Cấu hình</th><th>Hit@1</th><th>Hit@3</th><th>Hit@3 Strict</th><th>Context Hit</th>
          <th>Avg Chars</th><th>Median (ms)</th><th>Total (s)</th><th>Tokens (Embed / LLM)</th><th>Chi phí</th>
        </tr>
      </thead>
      <tbody>
        ${rows
          .map(
            (r) => `
          <tr class="${r.config === "hybrid_rerank_pack2400" ? "row-best" : ""}">
            <td><code>${esc(r.config)}</code></td>
            <td>${r.hit1}/${r.n} (${fmtPct(r.hit1_rate)})</td>
            <td>${r.hit3}/${r.n} (${fmtPct(r.hit3_rate)})</td>
            <td><strong>${r.hit3_strict}/${r.n}</strong></td>
            <td>${r.context_hit}/${r.n} (${fmtPct(r.context_hit_rate)})</td>
            <td>${r.avg_context_chars}</td>
            <td>${r.median_ms}</td>
            <td>${r.total_seconds}</td>
            <td>${r.embed_tokens} / ${r.llm_tokens}</td>
            <td>${fmtUsd(r.estimated_cost_usd)}</td>
          </tr>`
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function renderBasicAgent() {
  const split = state.basicAgentSplit;
  $$("#basic-agent-split .seg-btn").forEach((b) => b.classList.toggle("active", b.dataset.split === split));
  const pack = state.overview.basic.agent[split];
  const base = pack.baseline;
  const ch = pack.chosen;
  const ratio = (x) => (x ? `${x.n}/${x.N}` : "-");

  const metrics = [
    ["Provider errors", base.summary.provider_errors, ch.summary.provider_errors],
    ["Tool use", ratio(base.summary.tool_use), ratio(ch.summary.tool_use)],
    ["Grounded answer (mean 0–2)", `${base.summary.grounded_answer?.mean?.toFixed(4)}`, `${ch.summary.grounded_answer?.mean?.toFixed(4)}`],
    ["Citation validity", ratio(base.summary.citation_validity), ratio(ch.summary.citation_validity)],
    ["Answer citation coverage", ratio(base.summary.answer_citation_coverage), ratio(ch.summary.answer_citation_coverage)],
    ["Abstention (câu bẫy)", ratio(base.summary.abstention), ratio(ch.summary.abstention)],
    ["False abstention", ratio(base.summary.false_abstention), ratio(ch.summary.false_abstention)],
    ["Avg searches / query", base.stats.avg_searches, ch.stats.avg_searches],
    ["Median / P95 latency (ms)", `${base.stats.median_ms} / ${base.stats.p95_ms}`, `${ch.stats.median_ms} / ${ch.stats.p95_ms}`],
    ["Chi phí ước tính (USD)", fmtUsd(base.stats.total_estimated_cost_usd), fmtUsd(ch.stats.total_estimated_cost_usd)],
  ];

  $("#basic-agent-stats").innerHTML = `
    <table class="data-table">
      <thead><tr><th>Chỉ số (${esc(split)})</th><th>Baseline (dense_qdrant_800)</th><th>Chosen (hybrid_rerank_pack2400 + multi-tool)</th></tr></thead>
      <tbody>
        ${metrics.map(([k, v1, v2]) => `<tr><td><strong>${esc(k)}</strong></td><td>${esc(v1)}</td><td><strong>${esc(v2)}</strong></td></tr>`).join("")}
      </tbody>
    </table>
  `;

  const cases = (state.overview.basic.cases || []).filter((c) => c.split === split);
  if (!state.basicCaseId || !cases.some((c) => c.id === state.basicCaseId)) {
    state.basicCaseId = cases[0]?.id || null;
  }

  const chRevMap = Object.fromEntries((ch.reviews || []).map((r) => [r.query_id, r]));
  const baseTrMap = Object.fromEntries((base.traces || []).map((t) => [t.query_id, t]));
  const chTrMap = Object.fromEntries((ch.traces || []).map((t) => [t.query_id, t]));

  $("#basic-case-list").innerHTML = cases
    .map((c) => {
      const bt = baseTrMap[c.id];
      const ct = chTrMap[c.id];
      const bFail = c.answerable && bt?.status !== "answered";
      const badge = c.answerable
        ? bFail
          ? `<span class="pill pill-warn">Base trượt · Chosen OK</span>`
          : `<span class="pill pill-ok">Cả 2 đạt</span>`
        : `<span class="pill pill-indigo">Unanswerable Trap</span>`;
      return `
        <button type="button" class="case-item ${c.id === state.basicCaseId ? "active" : ""}" data-basic-case="${esc(c.id)}">
          <div class="case-item-top"><code>${esc(c.id)}</code>${badge}</div>
          <div class="case-item-q">${esc(c.question)}</div>
        </button>`;
    })
    .join("");

  const activeCase = cases.find((c) => c.id === state.basicCaseId);
  if (!activeCase) {
    $("#basic-case-detail").innerHTML = "";
    return;
  }
  const bt = baseTrMap[activeCase.id] || {};
  const ct = chTrMap[activeCase.id] || {};
  const cr = chRevMap[activeCase.id] || {};
  $("#basic-case-detail").innerHTML = `
    <div style="margin-bottom:10px;">
      <code>${esc(activeCase.id)}</code> · <strong>${activeCase.answerable ? "Answerable" : "Unanswerable Trap"}</strong>
      ${cr.answer_grade !== undefined ? `· <span class="pill pill-ok">Chosen Grade: ${cr.answer_grade}/2</span>` : ""}
    </div>
    <p style="margin:0 0 12px;font-weight:600;">${esc(activeCase.question)}</p>
    <div class="compare-grid">
      <div class="card" style="padding:12px;">
        <div class="answer-header"><strong>Baseline (800c)</strong> ${statusPill(bt.status)}</div>
        <div class="answer-body" style="font-size:13px;">${esc(bt.answer)}</div>
        <div class="telemetry-bar" style="font-size:11.5px;">
          <span>Cits: ${(bt.citations || []).length}</span>
          <span>${bt.elapsed_ms || 0} ms</span>
        </div>
        <h4 style="margin:10px 0 6px;font-size:12.5px;">Events</h4>
        <div class="trace-list">${renderTraceEvents(bt.events)}</div>
      </div>
      <div class="card" style="padding:12px;border-color:#a7f3d0;">
        <div class="answer-header"><strong>Chosen (2400c Hybrid Rerank)</strong> ${statusPill(ct.status)}</div>
        <div class="answer-body" style="font-size:13px;">${esc(ct.answer)}</div>
        <div class="telemetry-bar" style="font-size:11.5px;">
          <span>Cits: <code>${esc((ct.citations || []).join(", "))}</code></span>
          <span>${ct.elapsed_ms || 0} ms</span>
        </div>
        <h4 style="margin:10px 0 6px;font-size:12.5px;">Events</h4>
        <div class="trace-list">${renderTraceEvents(ct.events)}</div>
      </div>
    </div>
  `;
}

/* ==================== TAB 3: ADVANCED 232P ==================== */
function renderAdvancedTab() {
  const adv = state.overview.advanced;
  const ing = adv.ingest_manifest || {};
  const devSum = adv.splits.dev?.summary || {};
  const testSum = adv.splits.test?.summary || {};
  const testStats = adv.splits.test?.stats || {};

  $("#adv-kpi-strip").innerHTML = `
    <div class="kpi-card">
      <div class="kpi-label">FPT_BCTN_2025.pdf Ingestion</div>
      <div class="kpi-val">${ing.pages_processed || 232} trang · ${ing.total_chunks || 1243} chunks</div>
      <div class="kpi-sub">${ing.text_chunks || 835} text + ${ing.multimodal_chunks || 408} Vision · $${ing.total_cost_usd || 0}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Dev Set Tuning (v4 Locked)</div>
      <div class="kpi-val" style="color:var(--emerald);">${devSum.passed || 0}/${devSum.total || 0} (${fmtPct(devSum.accuracy)})</div>
      <div class="kpi-sub">Phân tách trang hoàn toàn (0 trang giao test)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Held-Out Test (Strict Audit)</div>
      <div class="kpi-val" style="color:var(--emerald);">${testSum.passed || 0}/${testSum.total || 0} (${fmtPct(testSum.accuracy)})</div>
      <div class="kpi-sub">answered: ${testStats.statuses?.answered ?? 0} · abstain: ${testStats.statuses?.abstain ?? 0} · errors: ${testStats.statuses?.error ?? 0}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Hiệu năng Test (n = 30)</div>
      <div class="kpi-val">${testStats.median_ms || 0} ms</div>
      <div class="kpi-sub">${testStats.avg_tool_calls || 0} tool calls/câu · ${fmtUsd(testStats.total_estimated_cost_usd)}</div>
    </div>
  `;

  const devNotes = {
    v1: "Budget 4500 chật & xếp :t trước :v → rớt trang thứ 2 ở adv_dev_mh02; 10/18 case phải retry citation.",
    v2: "Tăng budget=8000, ưu tiên :v + phạt đa trang → sửa mh02, retry giảm 10→1; còn lỗi bỏ inspect_page_image ở mh03.",
    v3: "Bắt buộc inspect_page_image cho card KPI/bảng → sửa mh03 nhưng Vision 150 DPI làm adv_dev_t02 (p057) abstain sai.",
    v4: "Live full 200 DPI + quadrant 240 DPI + không ghi đè prose :t hợp lệ → 18/18 (100%) trên dev, khóa pipeline.",
  };

  $("#adv-dev-iterations").innerHTML = `
    <table class="data-table">
      <thead>
        <tr><th>Phiên bản</th><th>Answered / Abstain</th><th>Retry Citation</th><th>Median (ms)</th><th>Chi phí <code>dev</code></th><th>Ghi chú thay đổi &amp; sửa lỗi</th></tr>
      </thead>
      <tbody>
        ${(adv.dev_iterations || [])
          .map(
            (it) => `
          <tr class="${it.version === "v4" ? "row-best" : ""}">
            <td><strong>${esc(it.version)}</strong></td>
            <td>${it.statuses?.answered || 0} / ${it.statuses?.abstain || 0}</td>
            <td><strong>${it.citation_retries}/18</strong></td>
            <td>${it.median_ms}</td>
            <td>${fmtUsd(it.total_estimated_cost_usd)}</td>
            <td>${esc(devNotes[it.version] || "")}</td>
          </tr>`
          )
          .join("")}
      </tbody>
    </table>
  `;

  renderAdvSplit();
}

function renderAdvSplit() {
  const split = state.advSplit;
  $$("#adv-split-switch .seg-btn").forEach((b) => b.classList.toggle("active", b.dataset.split === split));
  const pack = state.overview.advanced.splits[split] || {};
  const sum = pack.summary || {};
  const byCat = sum.by_category || {};

  $("#adv-category-pills").innerHTML = Object.entries(byCat)
    .map(([cat, v]) => {
      const ok = v.passed === v.total;
      return `
        <div class="cat-card">
          <div class="cat-name">${esc(cat)}</div>
          <div class="cat-score" style="color:${ok ? "var(--emerald)" : "var(--amber)"};">${v.passed}/${v.total}</div>
        </div>`;
    })
    .join("");

  const cats = ["all", "failed", ...Object.keys(byCat)];
  $("#adv-filter-bar").innerHTML = cats
    .map((f) => {
      const label = f === "all" ? `Tất cả (${(pack.cases || []).length})` : f === "failed" ? "Chỉ câu FAIL" : f;
      return `<button type="button" class="filter-btn ${state.advFilter === f ? "active" : ""}" data-adv-filter="${esc(f)}">${esc(label)}</button>`;
    })
    .join("");

  const revMap = Object.fromEntries((pack.reviews || []).map((r) => [r.query_id, r]));
  const trMap = Object.fromEntries((pack.traces || []).map((t) => [t.query_id, t]));
  const cases = (pack.cases || []).filter((c) => {
    const rev = revMap[c.id];
    if (state.advFilter === "failed") return rev && !rev.passed;
    if (state.advFilter !== "all") return c.category === state.advFilter;
    return true;
  });

  if (!state.advCaseId || !cases.some((c) => c.id === state.advCaseId)) {
    state.advCaseId = cases[0]?.id || null;
  }

  $("#adv-case-list").innerHTML = cases
    .map((c) => {
      const rev = revMap[c.id];
      const passBadge = rev
        ? rev.passed
          ? `<span class="pill pill-ok">PASS</span>`
          : `<span class="pill pill-err">FAIL</span>`
        : `<span class="pill pill-muted">N/A</span>`;
      const srcStr = (c.source_pages || []).length ? `tr.${c.source_pages.join(",")}` : "bẫy";
      return `
        <button type="button" class="case-item ${c.id === state.advCaseId ? "active" : ""}" data-adv-case="${esc(c.id)}">
          <div class="case-item-top">
            <span><code>${esc(c.id)}</code> · <small>${esc(c.category)} (${esc(srcStr)})</small></span>
            ${passBadge}
          </div>
          <div class="case-item-q">${esc(c.question)}</div>
        </button>`;
    })
    .join("");

  const ac = cases.find((c) => c.id === state.advCaseId);
  if (!ac) {
    $("#adv-case-detail").innerHTML = `<p class="meta-hint">Không có câu hỏi khớp bộ lọc.</p>`;
    return;
  }
  const tr = trMap[ac.id] || {};
  const rev = revMap[ac.id] || {};
  const allPages = Array.from(new Set([...(ac.source_pages || []), ...(rev.citation_pages || [])])).sort((a, b) => a - b);
  const thumbs = allPages
    .map(
      (p) => `
      <div class="thumb-card">
        <img src="/api/advanced/page/${p}" alt="FPT BCTN Trang ${p}" data-lightbox-src="/api/advanced/page/${p}?region=full" data-lightbox-cap="FPT BCTN 2025 — Trang ${p}" />
        <div class="thumb-cap">Trang ${p} ${(ac.source_pages || []).includes(p) ? "· Nguồn chuẩn" : "· Agent trích dẫn"}</div>
      </div>`
    )
    .join("");

  $("#adv-case-detail").innerHTML = `
    <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:8px;">
      <code>${esc(ac.id)}</code>
      <span class="pill pill-indigo">${esc(ac.category)}</span>
      ${statusPill(tr.status)}
      ${rev.passed !== undefined ? (rev.passed ? `<span class="pill pill-ok">Audit: PASS</span>` : `<span class="pill pill-err">Audit: FAIL</span>`) : ""}
      <span class="meta-hint">${tr.elapsed_ms || 0} ms · ${tr.llm_tokens || 0} tokens</span>
    </div>
    <p style="margin:0 0 8px;font-weight:700;">Câu hỏi: ${esc(ac.question)}</p>
    <div style="background:#fff;border:1px solid var(--border);border-radius:8px;padding:10px 12px;margin-bottom:10px;">
      <div style="font-size:12px;font-weight:700;color:var(--ink-soft);">ĐÁP ÁN KỲ VỌNG / KEY FACTS</div>
      <div style="font-size:13px;margin-top:3px;">${esc(ac.expected_answer || "ABSTAIN")} ${ac.key_facts ? `<br/><code>${esc(ac.key_facts.join(" | "))}</code>` : ""}</div>
    </div>
    <div style="background:#fff;border:1px solid ${rev.passed === false ? "#fecdd3" : "#a7f3d0"};border-radius:8px;padding:10px 12px;margin-bottom:10px;">
      <div style="font-size:12px;font-weight:700;color:var(--ink-soft);">CÂU TRẢ LỜI CỦA AGENT</div>
      <div class="answer-body" style="font-size:13.5px;margin-top:4px;">${esc(tr.answer)}</div>
      <div style="font-size:12px;margin-top:6px;color:var(--ink-muted);">
        <strong>Citations hợp lệ:</strong> <code>${esc((tr.citations || []).join(", ") || "Không có")}</code>
        ${rev.note ? `<br/><strong>Nhận xét kiểm chứng:</strong> ${esc(rev.note)}` : ""}
      </div>
    </div>
    <h4 style="margin:12px 0 6px;font-size:13px;">Trace công cụ (${(tr.events || []).length} bước)</h4>
    <div class="trace-list">${renderTraceEvents(tr.events)}</div>
    ${thumbs ? `<h4 style="margin:14px 0 6px;font-size:13px;">Ảnh trang gốc liên quan (nhấp để phóng to 200 DPI)</h4><div class="thumb-grid">${thumbs}</div>` : ""}
  `;
}

/* ==================== TAB 4: CORPUS EXPLORER ==================== */
function renderCorpusControls() {
  $$("#corpus-kb-switch .seg-btn").forEach((b) => b.classList.toggle("active", b.dataset.kb === state.corpusKb));
  if (state.corpusKb === "advanced") {
    $("#corpus-controls").innerHTML = `
      <button type="button" class="btn-secondary" id="corp-prev-page">← Trang trước</button>
      <label>Trang: <input type="number" id="corp-page-input" min="1" max="232" value="${state.corpusPage}" style="width:76px;" /> / 232</label>
      <button type="button" class="btn-secondary" id="corp-next-page">Trang sau →</button>
      <select id="corp-mod-sel">
        <option value="all" ${state.corpusModality === "all" ? "selected" : ""}>Cả Text + Vision (:t + :v)</option>
        <option value="multimodal" ${state.corpusModality === "multimodal" ? "selected" : ""}>Chỉ Vision (:v)</option>
        <option value="text" ${state.corpusModality === "text" ? "selected" : ""}>Chỉ PDF Text (:t)</option>
      </select>
      <input type="text" id="corp-search-input" value="${esc(state.corpusQuery)}" placeholder="Tìm từ khóa trên toàn 1.243 chunks (bỏ lọc trang nếu nhập)..." style="flex:1;min-width:220px;" />
      <button type="button" class="btn-primary" id="corp-search-btn" style="padding:7px 14px;">Lọc / Tìm</button>
    `;
  } else {
    $("#corpus-controls").innerHTML = `
      <select id="corp-src-sel">
        <option value="" ${state.corpusSource === "" ? "selected" : ""}>Cả 2 văn bản (210 chunks)</option>
        <option value="tt08-2021" ${state.corpusSource === "tt08-2021" ? "selected" : ""}>TT 08/2021 — Đại học (99 chunks)</option>
        <option value="tt18-2021" ${state.corpusSource === "tt18-2021" ? "selected" : ""}>TT 18/2021 — Tiến sĩ (111 chunks)</option>
      </select>
      <input type="text" id="corp-search-input" value="${esc(state.corpusQuery)}" placeholder="Tìm từ khóa trong quy chế..." style="flex:1;min-width:240px;" />
      <button type="button" class="btn-primary" id="corp-search-btn" style="padding:7px 14px;">Tìm chunk</button>
    `;
  }
}

async function loadCorpusData() {
  const q = state.corpusQuery.trim();
  const params = new URLSearchParams({ kb: state.corpusKb, limit: "50" });
  if (state.corpusKb === "advanced") {
    if (!q) params.set("page", String(state.corpusPage));
    if (state.corpusModality !== "all") params.set("modality", state.corpusModality);
  } else {
    if (state.corpusSource) params.set("source_id", state.corpusSource);
  }
  if (q) params.set("q", q);

  const resp = await fetch(`/api/chunks?${params.toString()}`);
  const data = await resp.json();
  const cards = (data.items || [])
    .map((ch) => {
      const modPill =
        ch.modality === "multimodal"
          ? `<span class="pill pill-sky">VISION :v</span>`
          : `<span class="pill pill-muted">TEXT</span>`;
      const jumpBtn =
        state.corpusKb === "advanced" && ch.page
          ? `<button type="button" class="btn-secondary" data-corpus-jump="${ch.page}">Mở ảnh Tr.${ch.page}</button>`
          : "";
      return `
        <div class="chunk-card">
          <div class="chunk-card-head">
            <span class="chunk-id">${esc(ch.id)}</span>
            <div style="display:flex;gap:5px;align-items:center;">
              ${modPill}
              <span class="pill pill-muted">${esc(ch.source_id || "")} · Tr.${ch.page}</span>
              ${jumpBtn}
            </div>
          </div>
          ${ch.section ? `<div class="chunk-section">${esc(ch.section)}</div>` : ""}
          <div class="chunk-text">${esc(ch.text)}</div>
        </div>`;
    })
    .join("");

  if (state.corpusKb === "advanced") {
    $("#corpus-body").innerHTML = `
      <div>
        <div class="page-frame">
          <img src="/api/advanced/page/${state.corpusPage}" alt="FPT BCTN 2025 Trang ${state.corpusPage}"
               data-lightbox-src="/api/advanced/page/${state.corpusPage}?region=full"
               data-lightbox-cap="FPT BCTN 2025 — Trang ${state.corpusPage}" />
        </div>
        <div class="meta-hint" style="margin-top:6px;">FPT BCTN 2025 · Trang ${state.corpusPage}/232 (nhấp vào ảnh để phóng to 200 DPI)</div>
      </div>
      <div>
        <div class="meta-hint" style="margin-bottom:8px;">Hiển thị ${data.items.length} / ${data.total} chunks ${q ? `(tìm "${esc(q)}")` : `của trang ${state.corpusPage}`}</div>
        <div class="chunk-stack">${cards || `<p class="meta-hint">Không có chunk phù hợp.</p>`}</div>
      </div>
    `;
  } else {
    $("#corpus-body").style.gridTemplateColumns = "1fr";
    $("#corpus-body").innerHTML = `
      <div>
        <div class="meta-hint" style="margin-bottom:8px;">Tìm thấy ${data.total} chunks MOET (hiển thị tối đa 50)</div>
        <div class="chunk-stack" style="max-height:660px;">${cards || `<p class="meta-hint">Không có chunk phù hợp.</p>`}</div>
      </div>
    `;
  }
  if (state.corpusKb === "advanced") {
    $("#corpus-body").style.gridTemplateColumns = "";
  }
}

/* ==================== BOOTSTRAP & EVENTS ==================== */
async function init() {
  try {
    const resp = await fetch("/api/overview");
    state.overview = await resp.json();
    renderStatusStrip(state.overview.status, state.overview);
    renderPresets(state.overview.presets);
    syncModeUI();
    renderBasicTab();
    renderAdvancedTab();
    renderCorpusControls();
    await loadCorpusData();
  } catch (err) {
    $("#status-strip").innerHTML = `<span class="pill pill-err">Lỗi kết nối server: ${esc(err.message)}</span>`;
  }
}

document.addEventListener("click", (e) => {
  const navBtn = e.target.closest(".nav-btn");
  if (navBtn) {
    switchTab(navBtn.dataset.tab);
    return;
  }

  const modeBtn = e.target.closest(".kb-switch .seg-btn");
  if (modeBtn) {
    state.kb = modeBtn.dataset.kb;
    state.mode = modeBtn.dataset.mode;
    syncModeUI();
    return;
  }

  const presetBtn = e.target.closest(".preset-chip");
  if (presetBtn && state.overview) {
    const p = state.overview.presets.find((x) => x.id === presetBtn.dataset.presetId);
    if (p) {
      state.kb = p.kb;
      state.mode = p.mode;
      syncModeUI();
      $("#ask-input").value = p.question;
      askLive(p.question);
    }
    return;
  }

  const sideBtn = e.target.closest(".side-tab-btn");
  if (sideBtn) {
    activateSideTab(sideBtn.dataset.sideTab);
    return;
  }

  const citBtn = e.target.closest("[data-cit-id]");
  if (citBtn) {
    jumpToCitation(citBtn.dataset.citId);
    return;
  }

  const jumpPageBtn = e.target.closest("[data-jump-page]");
  if (jumpPageBtn && state.liveResult) {
    state.livePage = Number(jumpPageBtn.dataset.jumpPage);
    renderLivePageViewer("advanced", state.liveResult.primary?.pages || [state.livePage]);
    activateSideTab("page");
    return;
  }

  if (e.target.id === "live-page-img" || e.target.id === "live-zoom-modal-btn") {
    const pg = state.livePage || 1;
    openLightbox(`/api/advanced/page/${pg}?region=${encodeURIComponent(state.liveRegion === "cached" ? "full" : state.liveRegion)}`, `FPT BCTN 2025 — Trang ${pg}`);
    return;
  }

  const lbTarget = e.target.closest("[data-lightbox-src]");
  if (lbTarget) {
    openLightbox(lbTarget.dataset.lightboxSrc, lbTarget.dataset.lightboxCap || "Xem ảnh trang");
    return;
  }

  if (e.target.id === "lightbox-close" || e.target.id === "lightbox") {
    closeLightbox();
    return;
  }

  const bRetBtn = e.target.closest("#basic-ret-split .seg-btn");
  if (bRetBtn) {
    state.basicRetSplit = bRetBtn.dataset.split;
    renderBasicRetrieval();
    return;
  }

  const bAgBtn = e.target.closest("#basic-agent-split .seg-btn");
  if (bAgBtn) {
    state.basicAgentSplit = bAgBtn.dataset.split;
    state.basicCaseId = null;
    renderBasicAgent();
    return;
  }

  const bCaseBtn = e.target.closest("[data-basic-case]");
  if (bCaseBtn) {
    state.basicCaseId = bCaseBtn.dataset.basicCase;
    renderBasicAgent();
    return;
  }

  const advSpBtn = e.target.closest("#adv-split-switch .seg-btn");
  if (advSpBtn) {
    state.advSplit = advSpBtn.dataset.split;
    state.advFilter = "all";
    state.advCaseId = null;
    renderAdvSplit();
    return;
  }

  const advFltBtn = e.target.closest("[data-adv-filter]");
  if (advFltBtn) {
    state.advFilter = advFltBtn.dataset.advFilter;
    state.advCaseId = null;
    renderAdvSplit();
    return;
  }

  const advCaseBtn = e.target.closest("[data-adv-case]");
  if (advCaseBtn) {
    state.advCaseId = advCaseBtn.dataset.advCase;
    renderAdvSplit();
    return;
  }

  const corpKbBtn = e.target.closest("#corpus-kb-switch .seg-btn");
  if (corpKbBtn) {
    state.corpusKb = corpKbBtn.dataset.kb;
    state.corpusQuery = "";
    renderCorpusControls();
    loadCorpusData();
    return;
  }

  if (e.target.id === "corp-prev-page" && state.corpusPage > 1) {
    state.corpusPage -= 1;
    state.corpusQuery = "";
    renderCorpusControls();
    loadCorpusData();
    return;
  }

  if (e.target.id === "corp-next-page" && state.corpusPage < 232) {
    state.corpusPage += 1;
    state.corpusQuery = "";
    renderCorpusControls();
    loadCorpusData();
    return;
  }

  if (e.target.id === "corp-search-btn") {
    state.corpusQuery = $("#corp-search-input")?.value || "";
    const pgInput = $("#corp-page-input");
    if (pgInput) state.corpusPage = Math.max(1, Math.min(232, Number(pgInput.value) || 1));
    loadCorpusData();
    return;
  }

  const corpJump = e.target.closest("[data-corpus-jump]");
  if (corpJump) {
    state.corpusPage = Number(corpJump.dataset.corpusJump);
    state.corpusQuery = "";
    renderCorpusControls();
    loadCorpusData();
  }
});

document.addEventListener("change", (e) => {
  if (e.target.id === "live-region-sel" && state.liveResult) {
    state.liveRegion = e.target.value;
    renderLivePageViewer("advanced", state.liveResult.primary?.pages || [state.livePage]);
  }
  if (e.target.id === "corp-mod-sel") {
    state.corpusModality = e.target.value;
    loadCorpusData();
  }
  if (e.target.id === "corp-src-sel") {
    state.corpusSource = e.target.value;
    loadCorpusData();
  }
  if (e.target.id === "corp-page-input") {
    state.corpusPage = Math.max(1, Math.min(232, Number(e.target.value) || 1));
    state.corpusQuery = "";
    loadCorpusData();
  }
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeLightbox();
});

$("#ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  askLive($("#ask-input").value);
});

$("#ask-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    askLive($("#ask-input").value);
  }
});

init();
