const form = document.querySelector("#simulation-form");
const slider = document.querySelector("#timeline");
const star = document.querySelector("#star");
const statusPill = document.querySelector("#status");
const chart = document.querySelector("#chart");
const ctx = chart.getContext("2d");
const hrDiagram = document.querySelector("#hr-diagram");
const hrCtx = hrDiagram.getContext("2d");
const validationMessage = document.querySelector("#validation-message");

let frames = [];

const fields = {
  mass: document.querySelector("#mass"),
  radius: document.querySelector("#radius"),
  centralTemperature: document.querySelector("#central-temperature"),
  centralPressure: document.querySelector("#central-pressure"),
  hydrogen: document.querySelector("#hydrogen"),
  helium: document.querySelector("#helium"),
  metals: document.querySelector("#metals"),
};

const output = {
  phase: document.querySelector("#phase"),
  age: document.querySelector("#age"),
  colorLabel: document.querySelector("#color-label"),
  temperature: document.querySelector("#temperature"),
  luminosity: document.querySelector("#luminosity"),
  radius: document.querySelector("#radius-output"),
  hrPosition: document.querySelector("#hr-position"),
};

function formatYears(years) {
  if (years >= 1e9) return `${(years / 1e9).toFixed(2)} billion years`;
  if (years >= 1e6) return `${(years / 1e6).toFixed(2)} million years`;
  if (years >= 1e3) return `${(years / 1e3).toFixed(2)} thousand years`;
  return `${years.toFixed(0)} years`;
}

function payloadFromForm() {
  return {
    mass_solar: Number(fields.mass.value),
    radius_solar: Number(fields.radius.value),
    central_temperature: Number(fields.centralTemperature.value),
    central_pressure: Number(fields.centralPressure.value),
    composition: {
      hydrogen: Number(fields.hydrogen.value),
      helium: Number(fields.helium.value),
      metals: Number(fields.metals.value),
    },
    radial_steps: 800,
    frames: 180,
  };
}

function setStatus(text) {
  statusPill.textContent = text;
}

function validatePayload(payload) {
  const issues = [];
  const options = [];
  const totalComposition =
    payload.composition.hydrogen + payload.composition.helium + payload.composition.metals;

  if (!Number.isFinite(payload.mass_solar) || payload.mass_solar < 0.1 || payload.mass_solar > 25) {
    issues.push("Mass is outside the supported range.");
    options.push("Choose mass between 0.1 and 25 solar masses.");
  }
  if (!Number.isFinite(payload.radius_solar) || payload.radius_solar < 0.1 || payload.radius_solar > 100) {
    issues.push("Radius is outside the supported range.");
    options.push("Choose radius between 0.1 and 100 solar radii.");
  }
  if (
    !Number.isFinite(payload.central_temperature) ||
    payload.central_temperature < 1e6 ||
    payload.central_temperature > 8e8
  ) {
    issues.push("Central temperature is outside the supported range.");
    options.push("Try central temperature values from 1e6 to 8e8 K.");
  }
  if (
    !Number.isFinite(payload.central_pressure) ||
    payload.central_pressure < 1e12 ||
    payload.central_pressure > 1e22
  ) {
    issues.push("Central pressure is outside the supported range.");
    options.push("Try pressure values such as 1e15, 2.45e17, 1e18, or 1e20 dyne cm^-2.");
  }
  if (Math.abs(totalComposition - 1) > 0.01) {
    issues.push("Composition fractions must add up to 1.");
    options.push("Use examples like H=0.70, He=0.28, Z=0.02 or H=0.73, He=0.25, Z=0.02.");
  }

  return { issues, options };
}

function showValidation(issues, options) {
  if (!issues.length) {
    validationMessage.hidden = true;
    validationMessage.textContent = "";
    return;
  }

  validationMessage.hidden = false;
  validationMessage.innerHTML = `
    <strong>Input check</strong>
    <span>${issues.join(" ")}</span>
    <span>Suggested options: ${options.join(" ")}</span>
  `;
}

function drawChart(currentIndex) {
  if (!frames.length) return;

  const width = chart.width;
  const height = chart.height;
  const pad = 28;
  ctx.clearRect(0, 0, width, height);

  ctx.strokeStyle = "#303746";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, pad);
  ctx.lineTo(pad, height - pad);
  ctx.lineTo(width - pad, height - pad);
  ctx.stroke();

  const maxLum = Math.max(...frames.map((frame) => frame.luminosity_solar));
  const maxTemp = Math.max(...frames.map((frame) => frame.effective_temperature));

  drawLine("luminosity_solar", maxLum, "#f6c567");
  drawLine("effective_temperature", maxTemp, "#62d6d2");

  const x = pad + (currentIndex / Math.max(frames.length - 1, 1)) * (width - 2 * pad);
  ctx.strokeStyle = "#eef3f8";
  ctx.beginPath();
  ctx.moveTo(x, pad);
  ctx.lineTo(x, height - pad);
  ctx.stroke();

  ctx.fillStyle = "#a7b2c0";
  ctx.font = "13px Segoe UI, sans-serif";
  ctx.fillText("Luminosity", pad + 8, pad + 16);
  ctx.fillText("Temperature", pad + 8, pad + 34);

  function drawLine(key, maxValue, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    frames.forEach((frame, index) => {
      const xPoint = pad + (index / Math.max(frames.length - 1, 1)) * (width - 2 * pad);
      const normalized = frame[key] / Math.max(maxValue, 1e-9);
      const yPoint = height - pad - normalized * (height - 2 * pad);
      if (index === 0) ctx.moveTo(xPoint, yPoint);
      else ctx.lineTo(xPoint, yPoint);
    });
    ctx.stroke();
  }
}

function drawHrDiagram(currentFrame) {
  const width = hrDiagram.width;
  const height = hrDiagram.height;
  const padLeft = 58;
  const padRight = 24;
  const padTop = 24;
  const padBottom = 48;
  const plotWidth = width - padLeft - padRight;
  const plotHeight = height - padTop - padBottom;

  hrCtx.clearRect(0, 0, width, height);
  hrCtx.fillStyle = "#0d1118";
  hrCtx.fillRect(0, 0, width, height);

  const gradient = hrCtx.createLinearGradient(padLeft, 0, width - padRight, 0);
  gradient.addColorStop(0, "rgba(150, 190, 255, 0.25)");
  gradient.addColorStop(0.45, "rgba(255, 245, 215, 0.18)");
  gradient.addColorStop(1, "rgba(255, 150, 96, 0.22)");
  hrCtx.fillStyle = gradient;
  hrCtx.fillRect(padLeft, padTop, plotWidth, plotHeight);

  drawMainSequence();
  drawAxes();

  const marker = hrPoint(currentFrame.effective_temperature, currentFrame.luminosity_solar);
  hrCtx.fillStyle = currentFrame.color.hex;
  hrCtx.shadowColor = currentFrame.color.hex;
  hrCtx.shadowBlur = 18;
  hrCtx.beginPath();
  hrCtx.arc(marker.x, marker.y, 9, 0, Math.PI * 2);
  hrCtx.fill();
  hrCtx.shadowBlur = 0;
  hrCtx.strokeStyle = "#ffffff";
  hrCtx.lineWidth = 2;
  hrCtx.stroke();

  output.hrPosition.textContent = `${Math.round(currentFrame.effective_temperature).toLocaleString()} K, ${currentFrame.luminosity_solar.toFixed(2)} L☉`;

  function hrPoint(temperature, luminosity) {
    const minLogT = Math.log10(3000);
    const maxLogT = Math.log10(30000);
    const minLogL = -4;
    const maxLogL = 6;
    const logT = Math.log10(Math.min(Math.max(temperature, 3000), 30000));
    const logL = Math.log10(Math.min(Math.max(luminosity, 1e-4), 1e6));
    const x = padLeft + ((maxLogT - logT) / (maxLogT - minLogT)) * plotWidth;
    const y = padTop + ((maxLogL - logL) / (maxLogL - minLogL)) * plotHeight;
    return { x, y };
  }

  function drawMainSequence() {
    hrCtx.strokeStyle = "rgba(255, 255, 255, 0.42)";
    hrCtx.lineWidth = 3;
    hrCtx.beginPath();
    const sequence = [
      [30000, 1e5],
      [18000, 1e3],
      [10000, 80],
      [5800, 1],
      [4300, 0.08],
      [3200, 0.002],
    ];
    sequence.forEach(([temperature, luminosity], index) => {
      const point = hrPoint(temperature, luminosity);
      if (index === 0) hrCtx.moveTo(point.x, point.y);
      else hrCtx.lineTo(point.x, point.y);
    });
    hrCtx.stroke();

    labelZone("Blue giants", 132, 80, "#9fc0ff");
    labelZone("Main sequence", 300, 190, "#f8f0ce");
    labelZone("Red giants", 520, 76, "#ffb184");
    labelZone("White dwarfs", 118, 340, "#dce8ff");
  }

  function drawAxes() {
    hrCtx.strokeStyle = "#303746";
    hrCtx.lineWidth = 1;
    hrCtx.strokeRect(padLeft, padTop, plotWidth, plotHeight);

    hrCtx.fillStyle = "#a7b2c0";
    hrCtx.font = "13px Segoe UI, sans-serif";
    hrCtx.fillText("hotter", padLeft, height - 16);
    hrCtx.fillText("cooler", width - padRight - 48, height - 16);
    hrCtx.fillText("Luminosity", 10, padTop + 18);
    hrCtx.fillText("Temperature", padLeft + plotWidth / 2 - 34, height - 16);
  }

  function labelZone(text, x, y, color) {
    hrCtx.fillStyle = color;
    hrCtx.font = "13px Segoe UI, sans-serif";
    hrCtx.fillText(text, x, y);
  }
}

function renderFrame(index) {
  const frame = frames[index];
  if (!frame) return;

  const starSize = Math.min(360, Math.max(88, 100 + Math.sqrt(frame.radius_solar) * 32));

  document.documentElement.style.setProperty("--star-color", frame.color.hex);
  document.documentElement.style.setProperty("--star-size", `${starSize}px`);
  star.style.backgroundColor = frame.color.hex;

  output.phase.textContent = frame.phase;
  output.age.textContent = formatYears(frame.age_years);
  output.colorLabel.textContent = frame.color.label;
  output.temperature.textContent = `${Math.round(frame.effective_temperature).toLocaleString()} K`;
  output.luminosity.textContent = `${frame.luminosity_solar.toFixed(2)} L☉`;
  output.radius.textContent = `${frame.radius_solar.toFixed(2)} R☉`;

  drawChart(index);
  drawHrDiagram(frame);
}

async function runSimulation() {
  setStatus("Running");
  const payload = payloadFromForm();
  const validation = validatePayload(payload);
  showValidation(validation.issues, validation.options);
  if (validation.issues.length) {
    setStatus("Check inputs");
    return;
  }

  const response = await fetch("/evolution", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await response.json();
  if (!response.ok) {
    setStatus("Error");
    showValidation([data.error || "Simulation failed."], ["Use the suggested ranges shown beside each input."]);
    return;
  }

  frames = data.timeline;
  slider.max = String(frames.length - 1);
  slider.value = "0";
  setStatus("Ready");
  showValidation([], []);
  renderFrame(0);
}

slider.addEventListener("input", () => {
  renderFrame(Number(slider.value));
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await runSimulation();
  } catch (error) {
    alert(error.message);
  }
});

runSimulation().catch((error) => {
  setStatus("Error");
  console.error(error);
});
