const scopeCanvas = document.getElementById("scope");
const scopePlay = document.getElementById("scope-play");
const scopeWave = document.getElementById("scope-wave");
const scopeSpec = document.getElementById("scope-spec");
const scopeStatus = document.getElementById("scope-status");

const channelNotes = {
  Cz: "EEG at the crown of the head. Electrical activity from the cortex.",
  F3: "EEG over the left frontal lobe, the same kind of signal as Cz from another site.",
  F4: "EEG over the right frontal lobe.",
  O1: "EEG over the left occipital lobe, toward the back of the head.",
  ECG: "A chest electrocardiogram. Each spike is one heartbeat.",
  PPG: "The pulse at the wrist, from blood volume. The wave rises with each beat.",
  EDA: "Skin conductance. It drifts as sweat-gland activity changes.",
  Temp: "Skin temperature at the wrist. It drifts slowly across the recording.",
};

const statusCopy = {
  wave:
    "EEG at four sites, chest ECG, pulse, skin conductance, and skin temperature. Each one loops on its own.",
  spectrogram:
    "Short-time spectra of the same recording. Frequency runs up each lane; the side shows each channel’s predominant band.",
};

const frequencyBands = [
  { id: "delta", mark: "δ", name: "Delta", lo: 0.5, hi: 4 },
  { id: "theta", mark: "θ", name: "Theta", lo: 4, hi: 8 },
  { id: "alpha", mark: "α", name: "Alpha", lo: 8, hi: 13 },
  { id: "beta", mark: "β", name: "Beta", lo: 13, hi: 30 },
  { id: "gamma", mark: "γ", name: "Gamma", lo: 30, hi: 40 },
];

const eegChannelIds = new Set(["cz", "f3", "f4", "o1"]);

function bandRangeText(band) {
  const lo = Number.isInteger(band.lo) ? String(band.lo) : band.lo.toFixed(1);
  const hi = Number.isInteger(band.hi) ? String(band.hi) : String(band.hi);
  return `${lo}–${hi} Hz`;
}

function parseColor(color) {
  const value = color || "#1c1915";
  if (value.startsWith("#")) {
    const hex = value.slice(1);
    const full =
      hex.length === 3
        ? hex
            .split("")
            .map((ch) => ch + ch)
            .join("")
        : hex;
    return {
      r: parseInt(full.slice(0, 2), 16),
      g: parseInt(full.slice(2, 4), 16),
      b: parseInt(full.slice(4, 6), 16),
    };
  }
  const match = value.match(/(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/);
  if (match) {
    return {
      r: Number(match[1]),
      g: Number(match[2]),
      b: Number(match[3]),
    };
  }
  return { r: 28, g: 25, b: 21 };
}

function fftRadix2(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i += 1) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wlenRe = Math.cos(ang);
    const wlenIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let wRe = 1;
      let wIm = 0;
      for (let j = 0; j < len / 2; j += 1) {
        const uRe = re[i + j];
        const uIm = im[i + j];
        const vRe = re[i + j + len / 2] * wRe - im[i + j + len / 2] * wIm;
        const vIm = re[i + j + len / 2] * wIm + im[i + j + len / 2] * wRe;
        re[i + j] = uRe + vRe;
        im[i + j] = uIm + vIm;
        re[i + j + len / 2] = uRe - vRe;
        im[i + j + len / 2] = uIm - vIm;
        const nextRe = wRe * wlenRe - wIm * wlenIm;
        wIm = wRe * wlenIm + wIm * wlenRe;
        wRe = nextRe;
      }
    }
  }
}

function buildSpectrogram(samples, hz) {
  const fftSize = 64;
  const hop = 4;
  const bins = fftSize / 2;
  const hann = new Float32Array(fftSize);
  for (let i = 0; i < fftSize; i += 1) {
    hann[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (fftSize - 1)));
  }

  const frameCount = Math.max(1, Math.floor((samples.length - fftSize) / hop) + 1);
  const linear = new Float32Array(frameCount * bins);
  const power = new Float32Array(frameCount * bins);
  const re = new Float32Array(fftSize);
  const im = new Float32Array(fftSize);
  let maxPower = 1e-12;
  const binHz = hz / fftSize;

  for (let frame = 0; frame < frameCount; frame += 1) {
    const start = frame * hop;
    re.fill(0);
    im.fill(0);
    for (let i = 0; i < fftSize; i += 1) {
      const sample = samples[Math.min(samples.length - 1, start + i)] || 0;
      re[i] = sample * hann[i];
    }
    fftRadix2(re, im);
    const base = frame * bins;
    for (let bin = 0; bin < bins; bin += 1) {
      const mag = re[bin] * re[bin] + im[bin] * im[bin];
      linear[base + bin] = mag;
      if (mag > maxPower) maxPower = mag;
    }
  }

  const logMax = Math.log10(maxPower + 1e-12);
  const logMin = logMax - 3.5;
  for (let i = 0; i < power.length; i += 1) {
    const log = Math.log10(linear[i] + 1e-12);
    power[i] = Math.max(0, Math.min(1, (log - logMin) / (logMax - logMin)));
  }

  return {
    power,
    linear,
    bins,
    hop,
    fftSize,
    binHz,
    frameCount,
    nyquist: hz / 2,
    duration: samples.length / hz,
  };
}

function predominantBand(spec, frame, radius = 0) {
  const start = Math.max(0, frame - radius);
  const end = Math.min(spec.frameCount - 1, frame + radius);
  const scores = frequencyBands.map((band) => {
    let total = 0;
    let count = 0;
    for (let f = start; f <= end; f += 1) {
      const base = f * spec.bins;
      for (let bin = 1; bin < spec.bins; bin += 1) {
        const freq = bin * spec.binHz;
        if (freq < band.lo || freq >= Math.min(band.hi, spec.nyquist + 1e-6)) continue;
        total += spec.linear[base + bin];
        count += 1;
      }
    }
    return { band, score: count ? total / count : 0 };
  });
  scores.sort((a, b) => b.score - a.score);
  return scores[0].band;
}

function startScope(recording) {
  const canvas = scopeCanvas;
  const names = document.getElementById("scope-names");
  const bandsEl = document.getElementById("scope-bands");
  const scopeEl = canvas.closest(".scope");
  const ctx = canvas.getContext("2d");
  const channels = recording.channels;
  const bandLabels = [];
  const monitor = canvas.closest(".monitor");
  const tip = document.createElement("p");
  tip.className = "tip";
  tip.hidden = true;
  tip.setAttribute("role", "tooltip");
  if (monitor) monitor.append(tip);

  let pinned = null;

  function placeTip(button) {
    if (!monitor) return;
    const bounds = monitor.getBoundingClientRect();
    const mark = button.getBoundingClientRect();
    tip.style.left = `${mark.right - bounds.left + 10}px`;
    tip.style.top = `${mark.top - bounds.top + mark.height / 2}px`;
    tip.style.transform = "translateY(-50%)";
    const placed = tip.getBoundingClientRect();
    if (placed.bottom > bounds.bottom - 8) {
      tip.style.top = `${bounds.height - tip.offsetHeight - 8}px`;
      tip.style.transform = "none";
    } else if (placed.top < bounds.top + 8) {
      tip.style.top = "8px";
      tip.style.transform = "none";
    }
  }

  function showTip(button) {
    tip.textContent =
      button.dataset.tip ||
      channelNotes[button.dataset.channel] ||
      button.dataset.channel ||
      "";
    tip.hidden = false;
    placeTip(button);
  }

  function hideTip() {
    if (pinned) return;
    tip.hidden = true;
  }

  function unpin() {
    if (!pinned) return;
    pinned.classList.remove("is-open");
    pinned.setAttribute("aria-expanded", "false");
    pinned = null;
    tip.hidden = true;
  }

  function wireInfo(info) {
    info.addEventListener("mouseenter", () => showTip(info));
    info.addEventListener("mouseleave", (event) => {
      if (event.relatedTarget !== tip) hideTip();
    });
    info.addEventListener("focus", () => showTip(info));
    info.addEventListener("blur", () => hideTip());
    info.addEventListener("click", (event) => {
      event.stopPropagation();
      if (pinned === info) {
        unpin();
        return;
      }
      unpin();
      pinned = info;
      info.classList.add("is-open");
      info.setAttribute("aria-expanded", "true");
      showTip(info);
    });
  }

  function makeInfoButton(color, label) {
    const info = document.createElement("button");
    info.type = "button";
    info.className = "info";
    info.style.color = color || "";
    info.setAttribute("aria-label", label);
    info.setAttribute("aria-expanded", "false");
    const mark = document.createElement("span");
    mark.setAttribute("aria-hidden", "true");
    mark.textContent = "i";
    info.append(mark);
    wireInfo(info);
    return info;
  }

  tip.addEventListener("mouseleave", () => hideTip());
  document.addEventListener("click", () => unpin());

  if (names) {
    names.style.setProperty("--lanes", String(channels.length));
    names.replaceChildren(
      ...channels.map((channel) => {
        const lane = document.createElement("span");
        lane.className = "lane";

        const label = document.createElement("span");
        label.className = "lane-name";
        label.textContent = channel.name;
        label.style.color = channel.color;

        const info = makeInfoButton(channel.color, `About ${channel.name}`);
        info.dataset.channel = channel.name;

        lane.append(label, info);
        return lane;
      })
    );
  }
  if (bandsEl) {
    bandsEl.style.setProperty("--lanes", String(channels.length));
    bandsEl.replaceChildren(
      ...channels.map((channel) => {
        const isEeg = eegChannelIds.has(channel.id);
        const label = document.createElement("div");
        label.className = isEeg ? "lane-band" : "lane-band lane-band-range";
        label.style.color = channel.color || "";

        const mark = document.createElement("strong");
        mark.textContent = "—";
        label.append(mark);

        const name = document.createElement("span");
        name.textContent = "";
        let info = null;
        if (isEeg) {
          const meta = document.createElement("span");
          meta.className = "band-meta";
          info = makeInfoButton(channel.color, `Frequency range for ${channel.name}`);
          meta.append(name, info);
          label.append(meta);
        } else {
          label.append(name);
        }

        bandLabels.push({ root: label, mark, name, info, isEeg });
        return label;
      })
    );
  }
  const hz = recording.hz;
  const windowSec = 6;
  let playing = !reduce;
  let mode = "wave";
  let playhead = 0;
  let last = performance.now();
  let pointer = null;
  const readout = document.getElementById("scope-read");
  const spectrograms = channels.map((channel) =>
    buildSpectrogram(channel.samples, hz)
  );
  const palette = channels.map((channel) => parseColor(channel.color));

  function setMode(next) {
    mode = next;
    const spectrogram = mode === "spectrogram";
    if (scopeWave) scopeWave.setAttribute("aria-pressed", spectrogram ? "false" : "true");
    if (scopeSpec) scopeSpec.setAttribute("aria-pressed", spectrogram ? "true" : "false");
    if (scopeStatus) scopeStatus.textContent = statusCopy[mode];
    if (scopeEl) scopeEl.classList.toggle("is-spectrogram", spectrogram);
    if (bandsEl) {
      bandsEl.hidden = !spectrogram;
      bandsEl.setAttribute("aria-hidden", spectrogram ? "false" : "true");
    }
  }

  function updateBands() {
    if (!bandLabels.length || mode !== "spectrogram") return;
    channels.forEach((channel, lane) => {
      const spec = spectrograms[lane];
      const frame = spectrogramAt(spec, playhead);
      const radius = Math.max(1, Math.round(hz / spec.hop / 2));
      const band = predominantBand(spec, frame, radius);
      const label = bandLabels[lane];
      const range = bandRangeText(band);
      if (label.isEeg) {
        if (label.mark.textContent === band.mark) return;
        label.mark.textContent = band.mark;
        label.name.textContent = band.name;
        if (label.info) {
          label.info.dataset.tip = range;
          label.info.setAttribute(
            "aria-label",
            `${band.name} frequency range for ${channel.name}`
          );
        }
        label.root.setAttribute(
          "aria-label",
          `${channel.name} predominant band ${band.name}`
        );
        return;
      }
      if (label.mark.textContent === range) return;
      label.mark.textContent = range;
      label.name.textContent = "";
      label.root.setAttribute(
        "aria-label",
        `${channel.name} predominant frequency ${range}`
      );
    });
  }

  function resize() {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  }

  function sampleAt(channel, time) {
    const span = channel.seconds || channel.samples.length / hz;
    const wrapped = ((time % span) + span) % span;
    const pos = wrapped * hz;
    const i = Math.floor(pos);
    const f = pos - i;
    const a = channel.samples[i] || 0;
    const b = channel.samples[Math.min(channel.samples.length - 1, i + 1)] || a;
    return a + (b - a) * f;
  }

  function formatValue(channel, sample) {
    const gain = channel.gain || 1;
    const offset = channel.offset || 0;
    const value = (sample / 100) * gain + offset;
    if (channel.unit === "°C") return `${value.toFixed(2)} °C`;
    const abs = Math.abs(value);
    const digits = abs >= 100 ? 0 : abs >= 10 ? 1 : 2;
    return value.toFixed(digits);
  }

  function spectrogramAt(spec, time) {
    const wrapped = ((time % spec.duration) + spec.duration) % spec.duration;
    const frame = Math.min(
      spec.frameCount - 1,
      Math.max(0, Math.floor((wrapped * hz) / spec.hop))
    );
    return frame;
  }

  function placeReadout() {
    if (!readout) return;
    if (!pointer) {
      readout.hidden = true;
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const laneHcss = rect.height / channels.length;
    const lane = Math.max(0, Math.min(channels.length - 1, Math.floor(pointer.y / laneHcss)));
    const channel = channels[lane];
    const time = playhead - windowSec + (pointer.x / rect.width) * windowSec;

    if (mode === "spectrogram") {
      const spec = spectrograms[lane];
      const yInLane = pointer.y - lane * laneHcss;
      const freq = Math.max(0, Math.min(spec.nyquist, (1 - yInLane / laneHcss) * spec.nyquist));
      const frame = spectrogramAt(spec, time);
      const bin = Math.min(
        spec.bins - 1,
        Math.max(0, Math.floor((freq / spec.nyquist) * spec.bins))
      );
      const level = Math.round(spec.power[frame * spec.bins + bin] * 100);
      readout.textContent = `${channel.name} ${freq.toFixed(1)} Hz · ${level}%`;
    } else {
      const sample = sampleAt(channel, time);
      readout.textContent = `${channel.name} ${formatValue(channel, sample)}`;
    }

    readout.hidden = false;
    readout.style.color = channel.color || "#1c1915";
    const gap = 12;
    let left = pointer.x + gap;
    let top = pointer.y - readout.offsetHeight - 10;
    if (left + readout.offsetWidth > rect.width - 6) {
      left = Math.max(6, pointer.x - readout.offsetWidth - gap);
    }
    if (top < 6) top = Math.min(rect.height - readout.offsetHeight - 6, pointer.y + 14);
    readout.style.left = `${left}px`;
    readout.style.top = `${top}px`;
  }

  function drawWave() {
    const width = canvas.width;
    const height = canvas.height;
    const laneH = height / channels.length;
    ctx.lineWidth = Math.max(1.1, width / 900);
    ctx.lineJoin = "round";
    ctx.lineCap = "round";

    channels.forEach((channel, lane) => {
      const mid = laneH * lane + laneH / 2;
      const amp = laneH * 0.36;
      ctx.strokeStyle = "rgba(28, 25, 21, 0.08)";
      ctx.beginPath();
      ctx.moveTo(0, mid);
      ctx.lineTo(width, mid);
      ctx.stroke();

      ctx.strokeStyle = channel.color || "#1c1915";
      ctx.beginPath();
      for (let x = 0; x < width; x += 2) {
        const time = playhead - windowSec + (x / width) * windowSec;
        const y = mid - (sampleAt(channel, time) / 100) * amp;
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    });

    if (!pointer) return;
    const rect = canvas.getBoundingClientRect();
    const x = (pointer.x / rect.width) * width;
    const laneHcss = rect.height / channels.length;
    const lane = Math.max(0, Math.min(channels.length - 1, Math.floor(pointer.y / laneHcss)));
    const channel = channels[lane];
    const mid = laneH * lane + laneH / 2;
    const amp = laneH * 0.36;
    const time = playhead - windowSec + (pointer.x / rect.width) * windowSec;
    const y = mid - (sampleAt(channel, time) / 100) * amp;

    ctx.strokeStyle = "rgba(28, 25, 21, 0.28)";
    ctx.lineWidth = Math.max(1, width / 900);
    ctx.beginPath();
    ctx.moveTo(x, laneH * lane);
    ctx.lineTo(x, laneH * (lane + 1));
    ctx.stroke();

    ctx.fillStyle = channel.color || "#1c1915";
    ctx.beginPath();
    ctx.arc(x, y, Math.max(3, width / 280), 0, Math.PI * 2);
    ctx.fill();
    placeReadout();
  }

  function drawSpectrogram() {
    const width = canvas.width;
    const height = canvas.height;
    const laneH = height / channels.length;
    const image = ctx.createImageData(width, height);
    const data = image.data;
    const step = Math.max(1, Math.floor(width / 360));

    channels.forEach((channel, lane) => {
      const spec = spectrograms[lane];
      const tint = palette[lane];
      const y0 = Math.floor(laneH * lane);
      const y1 = Math.floor(laneH * (lane + 1));
      const lanePixels = Math.max(1, y1 - y0);

      for (let x = 0; x < width; x += step) {
        const time = playhead - windowSec + (x / width) * windowSec;
        const frame = spectrogramAt(spec, time);
        const base = frame * spec.bins;
        const xEnd = Math.min(width, x + step);

        for (let y = y0; y < y1; y += 1) {
          const t = 1 - (y - y0 + 0.5) / lanePixels;
          const bin = Math.min(spec.bins - 1, Math.max(0, Math.floor(t * spec.bins)));
          const level = spec.power[base + bin];
          const alpha = Math.floor(18 + level * 210);
          for (let xx = x; xx < xEnd; xx += 1) {
            const i = (y * width + xx) * 4;
            data[i] = tint.r;
            data[i + 1] = tint.g;
            data[i + 2] = tint.b;
            data[i + 3] = alpha;
          }
        }
      }

      for (let x = 0; x < width; x += 1) {
        const i = (y0 * width + x) * 4;
        data[i] = 28;
        data[i + 1] = 25;
        data[i + 2] = 21;
        data[i + 3] = 28;
      }
    });

    ctx.putImageData(image, 0, 0);
    updateBands();

    if (!pointer) return;
    const rect = canvas.getBoundingClientRect();
    const x = (pointer.x / rect.width) * width;
    const laneHcss = rect.height / channels.length;
    const lane = Math.max(0, Math.min(channels.length - 1, Math.floor(pointer.y / laneHcss)));

    ctx.strokeStyle = "rgba(28, 25, 21, 0.34)";
    ctx.lineWidth = Math.max(1, width / 900);
    ctx.beginPath();
    ctx.moveTo(x, laneH * lane);
    ctx.lineTo(x, laneH * (lane + 1));
    ctx.stroke();
    placeReadout();
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (mode === "spectrogram") drawSpectrogram();
    else drawWave();
  }

  function tick(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    if (playing) playhead += dt;
    draw();
    requestAnimationFrame(tick);
  }

  scopePlay.addEventListener("click", () => {
    playing = !playing;
    scopePlay.textContent = playing ? "Pause" : "Play";
    last = performance.now();
  });

  if (scopeWave) {
    scopeWave.addEventListener("click", () => setMode("wave"));
  }
  if (scopeSpec) {
    scopeSpec.addEventListener("click", () => setMode("spectrogram"));
  }

  resize();
  window.addEventListener("resize", resize);
  canvas.addEventListener("pointermove", (event) => {
    const rect = canvas.getBoundingClientRect();
    pointer = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };
  });
  canvas.addEventListener("pointerdown", (event) => {
    const rect = canvas.getBoundingClientRect();
    pointer = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };
  });
  canvas.addEventListener("pointerleave", () => {
    pointer = null;
    if (readout) readout.hidden = true;
  });
  if (!playing) scopePlay.textContent = "Play";
  setMode("wave");
  requestAnimationFrame(tick);
}

if (scopeCanvas) {
  fetch("body.json?v=4")
    .then((response) => {
      if (!response.ok) throw new Error("missing recording");
      return response.json();
    })
    .then(startScope)
    .catch(() => {
      if (scopeStatus) scopeStatus.textContent = "The recording could not be loaded.";
    });
}

const roles = [
  "Software Engineer",
  "Electrical Engineer",
  "Embedded Systems",
  "Robotics Engineer",
  "DSP Engineer",
];
const role = document.getElementById("role");
const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

if (!reduce && role) {
  let index = 0;

  window.setInterval(() => {
    role.classList.add("is-out");

    window.setTimeout(() => {
      index = (index + 1) % roles.length;
      role.textContent = roles[index];
      role.classList.remove("is-out");
    }, 420);
  }, 3400);
}
