const chartColors = { actual: "#e8e9f3", xgboost: "#6c8cff", mlp: "#ff9f6c" };

async function fetchJSON(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return res.json();
}

function renderMetrics(metrics) {
  const container = document.getElementById("metrics");
  container.innerHTML = "";
  for (const [model, values] of Object.entries(metrics)) {
    for (const [metric, value] of Object.entries(values)) {
      const card = document.createElement("div");
      card.className = "metric-card";
      const suffix = metric === "MAPE" ? "%" : " units";
      card.innerHTML = `<h3>${model} · ${metric}</h3><div class="value">${value.toFixed(1)}${suffix}</div>`;
      container.appendChild(card);
    }
  }
}

function renderActualVsPredicted(rows) {
  const ctx = document.getElementById("actualVsPredictedChart");
  new Chart(ctx, {
    type: "line",
    data: {
      labels: rows.map((r) => r.date),
      datasets: [
        { label: "Actual", data: rows.map((r) => r.actual), borderColor: chartColors.actual, borderWidth: 2, pointRadius: 0 },
        { label: "XGBoost", data: rows.map((r) => r.xgboost), borderColor: chartColors.xgboost, borderDash: [5, 4], borderWidth: 2, pointRadius: 0 },
        { label: "PyTorch MLP", data: rows.map((r) => r.mlp), borderColor: chartColors.mlp, borderDash: [5, 4], borderWidth: 2, pointRadius: 0 },
      ],
    },
    options: baseChartOptions("Units sold (all categories)"),
  });
}

function renderFeatureImportance(rows) {
  const top = rows.slice(0, 12);
  const ctx = document.getElementById("featureImportanceChart");
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: top.map((r) => r.feature),
      datasets: [{ label: "Gain", data: top.map((r) => r.importance), backgroundColor: chartColors.xgboost }],
    },
    options: { ...baseChartOptions("Gain"), indexAxis: "y" },
  });
}

function baseChartOptions(yLabel) {
  return {
    responsive: true,
    scales: {
      x: { ticks: { color: "#9aa0c3", maxTicksLimit: 10 }, grid: { color: "#2a2f4a" } },
      y: { title: { display: true, text: yLabel, color: "#9aa0c3" }, ticks: { color: "#9aa0c3" }, grid: { color: "#2a2f4a" } },
    },
    plugins: { legend: { labels: { color: "#e8e9f3" } } },
  };
}

let forecastChartInstance = null;

function renderForecast(points) {
  const ctx = document.getElementById("forecastChart");
  if (forecastChartInstance) forecastChartInstance.destroy();
  forecastChartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels: points.map((p) => p.date),
      datasets: [
        { label: "Forecasted demand", data: points.map((p) => p.predicted_units_sold), borderColor: chartColors.xgboost, borderWidth: 2 },
      ],
    },
    options: baseChartOptions("Predicted units sold"),
  });
}

async function init() {
  const [metrics, actualVsPredicted, featureImportance, groups] = await Promise.all([
    fetchJSON("/api/metrics"),
    fetchJSON("/api/actual-vs-predicted"),
    fetchJSON("/api/feature-importance"),
    fetchJSON("/api/groups"),
  ]);

  renderMetrics(metrics);
  renderActualVsPredicted(actualVsPredicted);
  renderFeatureImportance(featureImportance);

  const categorySelect = document.getElementById("categorySelect");
  const subCategorySelect = document.getElementById("subCategorySelect");

  const byCategory = {};
  for (const g of groups) {
    (byCategory[g.category] ??= []).push(g.sub_category);
  }

  function populateSubCategories(category) {
    subCategorySelect.innerHTML = "";
    for (const sub of byCategory[category]) {
      const opt = document.createElement("option");
      opt.value = sub;
      opt.textContent = sub;
      subCategorySelect.appendChild(opt);
    }
  }

  for (const category of Object.keys(byCategory)) {
    const opt = document.createElement("option");
    opt.value = category;
    opt.textContent = category;
    categorySelect.appendChild(opt);
  }
  populateSubCategories(categorySelect.value);
  categorySelect.addEventListener("change", () => populateSubCategories(categorySelect.value));

  document.getElementById("forecastForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const discountValue = document.getElementById("discountInput").value;
    const body = {
      category: categorySelect.value,
      sub_category: subCategorySelect.value,
      horizon_days: Number(document.getElementById("horizonInput").value),
      discount_override_pct: discountValue === "" ? null : Number(discountValue),
    };
    const result = await fetchJSON("/api/forecast", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    renderForecast(result.points);
  });

  // Show an initial forecast on load.
  document.getElementById("forecastForm").dispatchEvent(new Event("submit"));
}

init().catch((err) => {
  console.error(err);
  document.body.insertAdjacentHTML("beforeend", `<p style="color:#ff6c6c;text-align:center">Failed to load dashboard: ${err.message}</p>`);
});
