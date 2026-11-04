"use strict";

const API_BASE = window.location.port === "8000" ? "" : "http://127.0.0.1:8000";
const elements = {};

let selectedFile = null;
let originalObjectUrl = null;
let resultUrl = null;
let resultReady = false;
let showingCovered = true;
let isRunning = false;

// ── Pipeline step definitions ────────────────────────────────────────────────
// Each step has a key matched against the backend status_text, a label, and an
// optional detail string shown when the step is active.
const IMAGE_STEPS = [
  { key: "upload",    label: "Upload",          match: /uploading/i },
  { key: "sam",       label: "Segment (SAM)",   match: /covering.*image|sam|segment/i },
  { key: "spark",     label: "Analyse (Spark)", match: /spark|analyse|tailoring/i },
  { key: "muse",      label: "Cover (Muse)",    match: /muse|cover|inpaint/i },
  { key: "done",      label: "Ready",           match: /ready/i },
];

const VIDEO_STEPS = [
  { key: "upload",    label: "Upload",             match: /uploading/i },
  { key: "extract",   label: "Extract Frames",     match: /reading frames/i },
  { key: "shots",     label: "Shot Detection",     match: /detecting shots/i },
  { key: "skin",      label: "Skin Pre-screen",    match: /checking frames for skin/i },
  { key: "comply",    label: "Compliance Gate",    match: /checking shot|checked shot/i },
  { key: "anchor",    label: "AI Edit (Anchors)",  match: /covering anchor/i },
  { key: "flow",      label: "Flow Propagation",   match: /propagating/i },
  { key: "encode",    label: "Encode Video",       match: /encoding/i },
  { key: "done",      label: "Ready",              match: /ready/i },
];

let currentSteps = IMAGE_STEPS;
let activeStepKey = null;
let completedStepKeys = new Set();

// ── Pipeline panel helpers ───────────────────────────────────────────────────
function buildPipelineSteps(steps) {
  elements.pipelineSteps.innerHTML = "";
  steps.forEach(({ key, label }) => {
    const li = document.createElement("li");
    li.className = "pipeline-step";
    li.dataset.key = key;
    li.innerHTML = `
      <span class="step-indicator" aria-hidden="true"></span>
      <span class="step-label">${label}</span>
      <span class="step-detail" id="step-detail-${key}"></span>
    `;
    elements.pipelineSteps.appendChild(li);
  });
}

function updatePipelineFromStatus(statusText, state) {
  if (!statusText) return;

  // Pick the active step by matching status_text against each step's regex
  const matched = currentSteps.find((s) => s.match.test(statusText));
  if (!matched) return;

  const newKey = matched.key;

  // Mark everything before this step as done
  let foundActive = false;
  currentSteps.forEach(({ key }) => {
    const li = elements.pipelineSteps.querySelector(`[data-key="${key}"]`);
    if (!li) return;
    if (key === newKey) {
      foundActive = true;
      li.className = "pipeline-step active";
      completedStepKeys.delete(key);
    } else if (!foundActive) {
      li.className = "pipeline-step done";
      completedStepKeys.add(key);
    } else {
      li.className = "pipeline-step pending";
    }
  });

  activeStepKey = newKey;

  // Write a short detail under the active step
  const detailEl = document.getElementById(`step-detail-${newKey}`);
  if (detailEl) {
    detailEl.textContent = statusText;
  }

  // Footer
  if (state === "ready") {
    markAllDone();
    elements.panelFooter.textContent = "all stages complete";
  } else if (state === "failed") {
    elements.panelFooter.textContent = "job failed";
    elements.panelFooter.classList.add("panel-footer--error");
  } else {
    elements.panelFooter.classList.remove("panel-footer--error");
    elements.panelFooter.textContent = statusText;
  }
}

function markAllDone() {
  elements.pipelineSteps.querySelectorAll(".pipeline-step").forEach((li) => {
    li.className = "pipeline-step done";
  });
}

function markStepSkipped(key, detail) {
  const step = elements.pipelineSteps.querySelector(`[data-key="${key}"]`);
  if (!step) return;
  step.className = "pipeline-step skipped";
  const detailEl = document.getElementById(`step-detail-${key}`);
  if (detailEl) detailEl.textContent = detail;
}

function resetPipeline(isVideo) {
  currentSteps = isVideo ? VIDEO_STEPS : IMAGE_STEPS;
  activeStepKey = null;
  completedStepKeys.clear();
  buildPipelineSteps(currentSteps);

  elements.panelMode.textContent = isVideo ? "video" : "image";
  elements.panelFooter.textContent = "waiting to start";
  elements.panelFooter.classList.remove("panel-footer--error");

  // Mark first step (upload) as active immediately
  const firstLi = elements.pipelineSteps.querySelector(".pipeline-step");
  if (firstLi) firstLi.className = "pipeline-step active";
}

function idlePipeline() {
  currentSteps = IMAGE_STEPS;
  buildPipelineSteps(currentSteps);
  elements.panelMode.textContent = "—";
  elements.panelFooter.textContent = "awaiting job";
  elements.panelFooter.classList.remove("panel-footer--error");
}

// ── Status / progress helpers ────────────────────────────────────────────────
function setStatus(text) {
  elements.statusReadout.textContent = text;
}

function setProgress(value) {
  const progress = Math.max(0, Math.min(100, Number(value) || 0));
  elements.progressFill.style.width = `${progress}%`;
  elements.progressTrack.setAttribute("aria-valuenow", String(Math.round(progress)));
}

function syncProcessButton() {
  elements.processButton.disabled =
    isRunning || !selectedFile || !elements.consentInput.checked;
}

function isVideoFile(file) {
  return file.type.startsWith("video/") || /\.(mp4|mov|webm)$/i.test(file.name);
}

function isSupportedFile(file) {
  return (
    isVideoFile(file) ||
    file.type.startsWith("image/") ||
    /\.(jpe?g|png|webp)$/i.test(file.name)
  );
}

function updatePlayerVisibility(kind) {
  const isVideo = kind === "video";
  elements.videoPlayer.hidden = !isVideo;
  elements.imagePlayer.hidden = isVideo;
  elements.dropPrompt.hidden = true;
}

function setPlayerSource(url, kind, currentTime = 0, wasPlaying = false) {
  updatePlayerVisibility(kind);
  if (kind === "video") {
    const player = elements.videoPlayer;
    const restorePlayback = () => {
      player.removeEventListener("loadedmetadata", restorePlayback);
      if (Number.isFinite(currentTime) && currentTime > 0) {
        player.currentTime = Math.min(currentTime, player.duration || currentTime);
      }
      if (wasPlaying) {
        player.play().catch((error) => console.warn("Playback could not resume:", error));
      }
    };
    player.addEventListener("loadedmetadata", restorePlayback);
    player.src = url;
    player.load();
  } else {
    elements.imagePlayer.src = url;
  }
}

function handleFileSelect(file) {
  if (!file) return;
  if (!isSupportedFile(file)) {
    setStatus("unsupported file — select video or image");
    return;
  }

  if (originalObjectUrl) URL.revokeObjectURL(originalObjectUrl);
  if (resultUrl) URL.revokeObjectURL(resultUrl);
  originalObjectUrl = URL.createObjectURL(file);
  resultUrl = null;
  selectedFile = file;
  resultReady = false;
  showingCovered = false;

  const kind = isVideoFile(file) ? "video" : "image";
  setPlayerSource(originalObjectUrl, kind);
  elements.fileReadout.textContent =
    `${file.name} · ${(file.size / 1048576).toFixed(1)} MB`;
  elements.viewToggle.hidden = true;
  elements.viewToggle.textContent = "view: original";
  elements.frameCounter.hidden = true;
  elements.frameCounter.textContent = "000 / 000";
  elements.tallyDot.classList.remove("active");
  elements.viewingFrame.classList.remove("is-processing");
  setProgress(0);
  setStatus("staged — ready to process");
  resetPipeline(kind === "video");
  syncProcessButton();
}

async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE}/api/health`);
    if (!response.ok) throw new Error(`Health check failed: ${response.status}`);
    elements.healthReadout.textContent = "link: online";
  } catch (error) {
    elements.healthReadout.textContent = "link: offline";
    console.error("Backend health check failed:", error);
  }
}

async function loadResult(jobId) {
  const kind = isVideoFile(selectedFile) ? "video" : "image";
  const currentTime = kind === "video" ? elements.videoPlayer.currentTime : 0;
  const wasPlaying = kind === "video" && !elements.videoPlayer.paused;
  const response = await fetch(`${API_BASE}/api/jobs/${jobId}/result`);
  if (!response.ok) {
    throw new Error(`Could not load result: ${response.status}`);
  }

  const resultBlob = await response.blob();
  if (resultUrl) URL.revokeObjectURL(resultUrl);
  resultUrl = URL.createObjectURL(resultBlob);
  resultReady = true;
  showingCovered = true;
  setPlayerSource(resultUrl, kind, currentTime, wasPlaying);
  elements.viewToggle.hidden = false;
  elements.viewToggle.textContent = "view: covered";
  setProgress(100);
}

async function pollJobStatus(jobId) {
  while (isRunning) {
    const response = await fetch(`${API_BASE}/api/jobs/${jobId}/status`);
    if (!response.ok) throw new Error(`Status request failed: ${response.status}`);
    const data = await response.json();

    if (typeof data.status_text === "string") {
      setStatus(data.status_text);
      updatePipelineFromStatus(data.status_text, data.state);
    }
    if (data.total_frames > 0) {
      elements.frameCounter.hidden = false;
      elements.frameCounter.textContent =
        `${String(data.current_frame || 0).padStart(3, "0")} / ${String(data.total_frames).padStart(3, "0")}`;
    }
    const calculatedProgress =
      data.total_frames > 0
        ? (data.current_frame / data.total_frames) * 100
        : 0;
    const reportedProgress = Number(data.progress);
    setProgress(
      Number.isFinite(reportedProgress) && reportedProgress > 0
        ? reportedProgress
        : calculatedProgress,
    );

    if (data.state === "ready" || data.state === "completed") {
      await loadResult(jobId);
      isRunning = false;
      elements.viewingFrame.classList.remove("is-processing");
      elements.tallyDot.classList.remove("active");
      setProgress(100);
      updatePipelineFromStatus("ready", "ready");
      if (isVideoFile(selectedFile)) {
        const anchorsEdited = Number(data.anchors_edited) || 0;
        const framesPropagated = Number(data.frames_propagated) || 0;
        const frameStatusCounts = data.frame_status_counts || {};
        const unchangedFrames = Object.entries(frameStatusCounts)
          .filter(([status]) => status !== "anchor_edited" && status !== "propagated")
          .reduce((total, [, count]) => total + (Number(count) || 0), 0);
        const screenedCompliant =
          (Number(frameStatusCounts.compliant_no_skin) || 0) +
          (Number(frameStatusCounts.skipped_compliant) || 0);
        const beforeAnchor = Number(frameStatusCounts.not_processed_before_anchor) || 0;
        elements.panelFooter.textContent =
          `${Number(data.shots_detected) || 0} shots detected`;
        if (anchorsEdited === 0) {
          setStatus("ready — no frames selected for AI edit");
          markStepSkipped("anchor", "no anchors selected");
          markStepSkipped("flow", "no edited anchors to propagate");
          elements.panelFooter.textContent = "AI edit skipped — no anchors selected";
        } else {
          setStatus(
            `ready — ${anchorsEdited} anchor${anchorsEdited === 1 ? "" : "s"} edited; ` +
            `${framesPropagated} propagated; ${unchangedFrames} unchanged ` +
            `(compliant ${screenedCompliant}, before anchor ${beforeAnchor})`,
          );
          if (framesPropagated === 0) {
            markStepSkipped("flow", "no frames required propagation");
          }
        }
      } else {
        setStatus("ready — output generated");
      }
      return;
    }

    if (data.state === "failed") {
      throw new Error(data.error || data.status_text || "Processing failed.");
    }

    await new Promise((resolve) => window.setTimeout(resolve, 800));
  }
}

async function startProcessing() {
  if (!selectedFile || !elements.consentInput.checked || isRunning) return;

  isRunning = true;
  syncProcessButton();
  elements.selectButton.disabled = true;
  elements.consentInput.disabled = true;
  elements.viewToggle.hidden = true;
  elements.viewingFrame.classList.remove("is-processing");
  void elements.viewingFrame.offsetWidth;
  elements.viewingFrame.classList.add("is-processing");
  elements.tallyDot.classList.add("active");
  setProgress(0);
  setStatus("uploading footage");
  updatePipelineFromStatus("uploading footage", "running");

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("consent", "true");

    const uploadResponse = await fetch(`${API_BASE}/api/upload`, {
      method: "POST",
      body: formData,
    });
    if (!uploadResponse.ok) {
      const error = await uploadResponse.json().catch(() => ({}));
      throw new Error(error.detail || `Upload failed: ${uploadResponse.status}`);
    }
    const { job_id: jobId } = await uploadResponse.json();
    if (!jobId) throw new Error("Upload response did not include a job ID.");

    const startResponse = await fetch(`${API_BASE}/api/jobs/${jobId}/start`, {
      method: "POST",
    });
    if (!startResponse.ok) {
      const error = await startResponse.json().catch(() => ({}));
      throw new Error(error.detail || `Could not start job: ${startResponse.status}`);
    }
    await pollJobStatus(jobId);
  } catch (error) {
    isRunning = false;
    elements.viewingFrame.classList.remove("is-processing");
    elements.tallyDot.classList.remove("active");
    setStatus(`error — ${error.message}`);
    updatePipelineFromStatus("failed", "failed");
    console.error("Media processing failed:", error);
  } finally {
    isRunning = false;
    elements.selectButton.disabled = false;
    elements.consentInput.disabled = false;
    syncProcessButton();
  }
}

function toggleResultView() {
  if (!resultReady || !resultUrl || !originalObjectUrl) return;

  const kind = isVideoFile(selectedFile) ? "video" : "image";
  const player = elements.videoPlayer;
  const currentTime = kind === "video" ? player.currentTime : 0;
  const wasPlaying = kind === "video" && !player.paused;
  showingCovered = !showingCovered;
  setPlayerSource(
    showingCovered ? resultUrl : originalObjectUrl,
    kind,
    currentTime,
    wasPlaying,
  );
  elements.viewToggle.textContent = showingCovered ? "view: covered" : "view: original";
}

function openFilePicker() {
  if (!isRunning) elements.fileInput.click();
}

function initialize() {
  elements.healthReadout  = document.getElementById("health-readout");
  elements.viewingFrame   = document.getElementById("viewing-frame");
  elements.fileInput      = document.getElementById("file-input");
  elements.dropPrompt     = document.getElementById("drop-prompt");
  elements.videoPlayer    = document.getElementById("video-player");
  elements.imagePlayer    = document.getElementById("image-player");
  elements.viewToggle     = document.getElementById("view-toggle");
  elements.progressTrack  = document.getElementById("progress-track");
  elements.progressFill   = document.getElementById("progress-fill");
  elements.fileReadout    = document.getElementById("file-readout");
  elements.statusReadout  = document.getElementById("status-readout");
  elements.frameCounter   = document.getElementById("frame-counter");
  elements.tallyDot       = document.getElementById("tally-dot");
  elements.consentInput   = document.getElementById("consent-input");
  elements.selectButton   = document.getElementById("select-btn");
  elements.processButton  = document.getElementById("process-btn");
  // Pipeline panel
  elements.pipelinePanel  = document.getElementById("pipeline-panel");
  elements.pipelineSteps  = document.getElementById("pipeline-steps");
  elements.panelMode      = document.getElementById("panel-mode");
  elements.panelFooter    = document.getElementById("panel-footer");

  elements.fileInput.addEventListener("change", (event) => {
    handleFileSelect(event.target.files[0]);
    event.target.value = "";
  });
  elements.selectButton.addEventListener("click", openFilePicker);
  elements.processButton.addEventListener("click", startProcessing);
  elements.consentInput.addEventListener("change", syncProcessButton);
  elements.viewToggle.addEventListener("click", toggleResultView);

  elements.viewingFrame.addEventListener("click", (event) => {
    if (
      event.target === elements.viewToggle ||
      event.target === elements.videoPlayer ||
      isRunning
    ) {
      return;
    }
    if (elements.videoPlayer.hidden || elements.videoPlayer.paused) openFilePicker();
  });
  elements.viewingFrame.addEventListener("keydown", (event) => {
    if (
      (event.key === "Enter" || event.key === " ") &&
      event.target === elements.viewingFrame
    ) {
      event.preventDefault();
      openFilePicker();
    }
  });

  elements.viewingFrame.addEventListener("dragover", (event) => {
    event.preventDefault();
    if (!isRunning) elements.viewingFrame.classList.add("drag-over");
  });
  elements.viewingFrame.addEventListener("dragleave", (event) => {
    if (!elements.viewingFrame.contains(event.relatedTarget)) {
      elements.viewingFrame.classList.remove("drag-over");
    }
  });
  elements.viewingFrame.addEventListener("drop", (event) => {
    event.preventDefault();
    elements.viewingFrame.classList.remove("drag-over");
    if (!isRunning) handleFileSelect(event.dataTransfer.files[0]);
  });

  window.addEventListener("beforeunload", () => {
    if (originalObjectUrl) URL.revokeObjectURL(originalObjectUrl);
    if (resultUrl) URL.revokeObjectURL(resultUrl);
  });

  idlePipeline();
  checkHealth();
  syncProcessButton();
}

document.addEventListener("DOMContentLoaded", initialize);
