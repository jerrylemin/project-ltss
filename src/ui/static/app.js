(function () {
  "use strict";

  const IDS = {
    body: "body",
    menuToggle: "menu-toggle",
    sidebarNav: "sidebar-nav",
    reloadData: "reload-data",
    benchmarkBadge: "benchmark-status-badge",
    graphBadge: "graph-status-badge",
    overviewStatus: "overview-status",
    overviewEmpty: "overview-empty-state",
    benchmarkEmpty: "benchmark-empty-state",
    benchmarkError: "benchmark-error",
    benchmarkTable: "benchmark-table",
    benchmarkTableBody: "benchmark-table-body",
    checklist: "checklist",
    presentationOpen: "presentation-open",
    presentationClose: "presentation-close",
    presentationPerformance: "presentation-performance",
    presentationStatus: "presentation-status"
  };

  const SECTION_IDS = {
    overview: "section-overview",
    architecture: "section-architecture",
    "data-structures": "section-data-structures",
    "algorithm-comparison": "section-algorithm-comparison",
    "push-pull": "section-push-pull",
    benchmarks: "section-benchmarks",
    verification: "section-verification"
  };

  const API = {
    benchmark: "/api/benchmark",
    graphs: "/api/graphs",
    status: "/api/status",
    shfl: "/api/shfl_check"
  };

  const state = {
    benchmark: null,
    graphs: null,
    status: null,
    shfl: null
  };

  function byId(id) {
    return document.getElementById(id);
  }

  function setBadge(element, text, kind) {
    element.textContent = text;
    element.className = "badge badge-" + kind;
  }

  function formatValue(value) {
    if (value === null || value === undefined) {
      return "\u2014";
    }
    const text = String(value).trim();
    if (text === "" || text.toLowerCase() === "nan") {
      return "\u2014";
    }
    return text;
  }

  function formatNumber(value, digits) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
      return "\u2014";
    }
    return number.toFixed(digits);
  }

  function formatNumberWithSuffix(value, digits, suffix) {
    const number = formatNumber(value, digits);
    return number === "\u2014" ? number : number + suffix;
  }

  async function getJson(url) {
    const response = await fetch(url, { headers: { "Accept": "application/json" } });
    if (!response.ok) {
      throw new Error("HTTP " + response.status);
    }
    return response.json();
  }

  function groupBenchmarkRows(rows) {
    const byGraph = new Map();
    rows.forEach(function (row) {
      const graph = row.graph_name || row.graph || "unknown";
      if (!byGraph.has(graph)) {
        byGraph.set(graph, { graph: graph, versions: {}, raw: [] });
      }
      const group = byGraph.get(graph);
      group.raw.push(row);
      if (row.version) {
        group.versions[row.version] = row;
      } else {
        group.wide = row;
      }
    });
    return Array.from(byGraph.values());
  }

  function bestGpuRow(group) {
    const names = ["gpu_v1", "gpu_v2", "gpu_v3_pull", "gpu_v3_push"];
    let best = null;
    names.forEach(function (name) {
      const row = group.versions[name];
      const time = row ? Number(row.convergence_time_s || row.best_gpu_time_s) : NaN;
      if (Number.isFinite(time) && (!best || time < best.time)) {
        best = { name: name, row: row, time: time };
      }
    });
    return best;
  }

  function normalizedRows(rows) {
    return groupBenchmarkRows(rows).map(function (group) {
      const cpu = group.versions.cpu_numpy || group.versions.cpu || group.wide || {};
      const v1 = group.versions.gpu_v1 || group.wide || {};
      const v2 = group.versions.gpu_v2 || group.wide || {};
      const pull = group.versions.gpu_v3_pull || group.wide || {};
      const push = group.versions.gpu_v3_push || group.wide || {};
      const best = bestGpuRow(group);
      const bestRow = best ? best.row : group.wide || {};
      const l1 = bestRow.relative_l1_vs_scipy || bestRow.scipy_l1_error;
      const tolerancePass = bestRow.tolerance_pass !== undefined
        ? parseBoolean(bestRow.tolerance_pass)
        : Number(l1) <= 1e-6;
      return {
        graph: group.graph,
        nodes: cpu.n_nodes || cpu.nodes || bestRow.n_nodes || bestRow.nodes,
        edges: cpu.n_edges || cpu.edges || bestRow.n_edges || bestRow.edges,
        cpuTime: cpu.convergence_time_s || cpu.cpu_time_s,
        v1Time: v1.convergence_time_s || v1.v1_time_s,
        v2Time: v2.convergence_time_s || v2.v2_time_s,
        pullTime: pull.convergence_time_s || pull.v3_pull_time_s,
        pushTime: push.convergence_time_s || push.v3_push_time_s,
        bestGpu: best ? best.name + " (" + formatNumber(best.time, 6) + "s)" : bestRow.best_gpu_time_s,
        speedup: bestRow.speedup_vs_cpu,
        l1: l1,
        spearman: bestRow.spearman_vs_scipy || bestRow.spearman_top1000,
        iterations: bestRow.iterations,
        tolerancePass: tolerancePass,
        youtubeUnder5: bestRow.target_under_5s !== undefined
          ? parseBoolean(bestRow.target_under_5s)
          : parseBoolean(bestRow.youtube_under_5s)
      };
    });
  }

  function parseBoolean(value) {
    if (value === true || value === 1) {
      return true;
    }
    if (value === false || value === 0) {
      return false;
    }
    const text = String(value).trim().toLowerCase();
    if (["true", "1", "yes", "pass"].includes(text)) {
      return true;
    }
    if (["false", "0", "no", "fail"].includes(text)) {
      return false;
    }
    return null;
  }

  function badgeHtml(value, notApplicable) {
    if (notApplicable) {
      return '<span class="badge badge-pending">N/A</span>';
    }
    if (value === true) {
      return '<span class="badge badge-pass">PASS</span>';
    }
    if (value === false) {
      return '<span class="badge badge-fail">FAIL</span>';
    }
    return '<span class="badge badge-pending">N/A</span>';
  }

  function renderBenchmark() {
    const table = byId(IDS.benchmarkTable);
    const tbody = byId(IDS.benchmarkTableBody);
    const empty = byId(IDS.benchmarkEmpty);
    const error = byId(IDS.benchmarkError);
    tbody.innerHTML = "";
    error.classList.add("hidden");

    if (!state.benchmark || state.benchmark.status === "missing") {
      empty.classList.remove("hidden");
      table.classList.add("hidden");
      return;
    }

    if (state.benchmark.status === "error") {
      empty.classList.add("hidden");
      table.classList.add("hidden");
      error.textContent = "Benchmark CSV error: " + state.benchmark.error;
      error.classList.remove("hidden");
      return;
    }

    empty.classList.add("hidden");
    table.classList.remove("hidden");
    normalizedRows(state.benchmark.rows).forEach(function (row) {
      const tr = document.createElement("tr");
      const isYoutube = row.graph.toLowerCase().includes("youtube");
      if (isYoutube) {
        tr.className = "highlight-row";
      }
      const ytCell = badgeHtml(row.youtubeUnder5, !isYoutube);
      tr.innerHTML = [
        "<th>" + formatValue(row.graph) + "</th>",
        "<td>" + formatValue(row.nodes) + "</td>",
        "<td>" + formatValue(row.edges) + "</td>",
        "<td>" + formatNumber(row.cpuTime, 6) + "</td>",
        "<td>" + formatNumber(row.v1Time, 6) + "</td>",
        "<td>" + formatNumber(row.v2Time, 6) + "</td>",
        "<td>" + formatNumber(row.pullTime, 6) + "</td>",
        "<td>" + formatNumber(row.pushTime, 6) + "</td>",
        "<td>" + formatValue(row.bestGpu) + "</td>",
        "<td>" + formatNumberWithSuffix(row.speedup, 3, "x") + "</td>",
        "<td>" + formatValue(row.l1) + "</td>",
        "<td>" + formatNumber(row.spearman, 3) + "</td>",
        "<td>" + formatValue(row.iterations) + "</td>",
        "<td>" + badgeHtml(row.tolerancePass, false) + "</td>",
        "<td>" + ytCell + "</td>"
      ].join("");
      tbody.appendChild(tr);
    });
  }

  function renderStatus() {
    const overview = byId(IDS.overviewStatus);
    const empty = byId(IDS.overviewEmpty);
    const benchmark = state.status ? state.status.benchmark_csv : null;
    const graphFiles = state.status ? state.status.graph_files : null;
    const files = graphFiles && graphFiles.files && graphFiles.files.length
      ? graphFiles.files.join(", ")
      : "none found";

    if (!benchmark) {
      return;
    }

    if (benchmark.present) {
      setBadge(byId(IDS.benchmarkBadge), "Benchmark present", "pass");
      empty.classList.add("hidden");
    } else {
      setBadge(byId(IDS.benchmarkBadge), "Benchmark missing", "fail");
      empty.classList.remove("hidden");
    }

    setBadge(
      byId(IDS.graphBadge),
      graphFiles && graphFiles.files && graphFiles.files.length ? "Graphs detected" : "No graph files",
      graphFiles && graphFiles.files && graphFiles.files.length ? "pass" : "pending"
    );

    overview.innerHTML = [
      '<div class="status-pill"><span>Benchmark CSV</span><strong>' + (benchmark.present ? "present" : "missing") + "</strong></div>",
      '<div class="status-pill"><span>Graph files detected</span><strong>' + files + "</strong></div>",
      '<div class="status-pill"><span>Last benchmark run</span><strong>' + formatValue(benchmark.last_modified) + "</strong></div>"
    ].join("");
  }

  function renderChecklist() {
    const target = byId(IDS.checklist);
    const checklist = state.status ? state.status.checklist : {};
    const shfl = state.shfl || { found: false };
    const items = [
      {
        label: "CPU baseline on roadNet-CA",
        detected: Boolean(checklist.roadnet_cpu_baseline_detected),
        status: checklist.roadnet_cpu_baseline_detected ? "detected" : "not detected"
      },
      {
        label: "pytest tests pass",
        detected: null,
        status: "Run: python -m pytest tests/ -v"
      },
      {
        label: "Full benchmark completed",
        detected: Boolean(checklist.full_benchmark_completed),
        status: checklist.full_benchmark_completed ? "detected" : "not detected"
      },
      {
        label: "com-youtube row in CSV",
        detected: Boolean(checklist.youtube_row_detected),
        status: checklist.youtube_row_detected ? "detected" : "not detected"
      },
      {
        label: "__shfl_down_sync exists in source",
        detected: Boolean(shfl.found),
        status: shfl.found ? "detected: " + shfl.matches.join(", ") : "not detected"
      },
      {
        label: "No host copies in V1 loop",
        detected: null,
        status: "Manual review required: src/kernels/v1*.cu or src/gpu/pagerank_v1.py"
      }
    ];

    target.innerHTML = "";
    items.forEach(function (item) {
      const div = document.createElement("div");
      const detected = item.detected === true;
      div.className = "check-item" + (detected ? " detected" : "");
      const badge = item.detected === null
        ? '<span class="badge badge-pending">MANUAL</span>'
        : badgeHtml(detected, false);
      div.innerHTML = '<span class="check-icon">' + (detected ? "\u2713" : "\u25cb") + '</span><div><strong>' + item.label + '</strong><p>' + item.status + '</p></div>' + badge;
      target.appendChild(div);
    });

    const passCount = items.filter(function (item) { return item.detected === true; }).length;
    const failCount = items.filter(function (item) { return item.detected === false; }).length;
    byId(IDS.presentationStatus).textContent = passCount + " detected, " + failCount + " not detected, 2 manual checks.";
  }

  function renderPresentationPerformance() {
    const target = byId(IDS.presentationPerformance);
    if (!state.benchmark || state.benchmark.status !== "ok") {
      target.textContent = "pending measurement";
      return;
    }
    const rows = normalizedRows(state.benchmark.rows);
    const youtube = rows.find(function (row) {
      return row.graph.toLowerCase().includes("youtube");
    });
    if (!youtube) {
      target.textContent = "pending measurement";
      return;
    }
    target.textContent = "com-youtube best GPU: " + formatValue(youtube.bestGpu) + "; speedup: " + formatNumberWithSuffix(youtube.speedup, 3, "x") + "; target under 5 seconds: " + (youtube.youtubeUnder5 ? "PASS" : "FAIL") + ".";
  }

  async function loadAllData() {
    const results = await Promise.allSettled([
      getJson(API.benchmark),
      getJson(API.graphs),
      getJson(API.status),
      getJson(API.shfl)
    ]);

    if (results[0].status === "fulfilled") {
      state.benchmark = results[0].value;
    } else {
      byId(IDS.benchmarkError).textContent = "Failed to fetch benchmark data: " + results[0].reason.message;
      byId(IDS.benchmarkError).classList.remove("hidden");
    }
    if (results[1].status === "fulfilled") {
      state.graphs = results[1].value;
    }
    if (results[2].status === "fulfilled") {
      state.status = results[2].value;
    }
    if (results[3].status === "fulfilled") {
      state.shfl = results[3].value;
    }

    renderStatus();
    renderBenchmark();
    renderChecklist();
    renderPresentationPerformance();
  }

  function activateSection(sectionName) {
    Object.keys(SECTION_IDS).forEach(function (key) {
      const section = byId(SECTION_IDS[key]);
      section.classList.toggle("active", key === sectionName);
    });
    document.querySelectorAll(".nav-link[data-section]").forEach(function (link) {
      link.classList.toggle("active", link.getAttribute("data-section") === sectionName);
    });
    document.body.classList.remove("nav-open");
    byId(IDS.menuToggle).setAttribute("aria-expanded", "false");
  }

  function bindEvents() {
    document.querySelectorAll(".nav-link[data-section]").forEach(function (link) {
      link.addEventListener("click", function () {
        activateSection(link.getAttribute("data-section"));
      });
    });
    document.querySelectorAll("[data-reload]").forEach(function (button) {
      button.addEventListener("click", loadAllData);
    });
    byId(IDS.reloadData).addEventListener("click", loadAllData);
    byId(IDS.menuToggle).addEventListener("click", function () {
      const open = document.body.classList.toggle("nav-open");
      byId(IDS.menuToggle).setAttribute("aria-expanded", String(open));
    });
    byId(IDS.presentationOpen).addEventListener("click", function () {
      document.body.classList.add("presentation-active");
    });
    byId(IDS.presentationClose).addEventListener("click", function () {
      document.body.classList.remove("presentation-active");
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.body.classList.add("js-enabled");
    bindEvents();
    loadAllData();
  });
}());
