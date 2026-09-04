/**
 * frontend/app.js - Modern Roboflow-Style Canvas & Annotation Controller
 * =======================================================================
 * Features:
 * - HTML5 Canvas with Pan/Zoom & High-DPI support
 * - Deterministic 4-Corner OBB rendering with Top-Left Anchor & Clockwise Winding
 * - Interactive Corner Dragging, Translation, and Rotation Handle
 * - Real-time AI Labeling (Waterfall), Run All (Comparison), and SAM Refinement
 * - Differential History & Model Telemetry
 * - Batch Auto-Labeling Engine with Pause/Resume/Cancel
 * - Comprehensive Keyboard Shortcuts
 */

// --- STATE DEFINITIONS ---
const state = {
  datasets: [],
  activeDatasetId: null,
  images: [],
  activeImageIndex: -1,
  activeImageMeta: null,
  imageObj: null,
  
  boxes: [],            // Active rendered boxes: [{xyxy, corners, confidence, class_id, class_name, model_source, needs_review, review_reasons, attributes}]
  rawPredictions: null, // Untouched model outputs from predictions/
  selectedBoxIndex: -1,
  hoveredBoxIndex: -1,
  hoveredHandle: null,  // null, 'c0','c1','c2','c3','rotate','body'
  
  toolMode: 'select',   // 'select' or 'draw'
  isDrawing: false,
  drawStart: null,
  
  transform: { scale: 1.0, tx: 0, ty: 0 },
  isPanning: false,
  panStart: { x: 0, y: 0 },
  
  isDraggingHandle: false,
  dragTarget: null,
  
  filterStatus: 'all',
  layerVisibility: { yolo: true, dino: true, sam: true },
  
  batchPollInterval: null,
  activeDevice: 'AUTO',
};

// --- DOM ELEMENTS ---
const elements = {
  datasetSelect: document.getElementById('dataset-select'),
  btnNewDataset: document.getElementById('btn-new-dataset'),
  btnImportImages: document.getElementById('btn-import-images'),
  btnAiLabel: document.getElementById('btn-ai-label'),
  btnRunAll: document.getElementById('btn-run-all'),
  btnRefineSam: document.getElementById('btn-refine-sam'),
  btnBatchModal: document.getElementById('btn-batch-modal'),
  btnExportDataset: document.getElementById('btn-export-dataset'),
  btnShortcuts: document.getElementById('btn-shortcuts'),
  
  hardwareLabel: document.getElementById('hardware-label'),
  deviceSelect: document.getElementById('device-select'),
  
  imageCountBadge: document.getElementById('image-count-badge'),
  statVerified: document.getElementById('stat-verified'),
  statReview: document.getElementById('stat-review'),
  statUnlabeled: document.getElementById('stat-unlabeled'),
  imageGallery: document.getElementById('image-gallery'),
  
  viewport: document.getElementById('viewport'),
  canvas: document.getElementById('annotation-canvas'),
  canvasSpinner: document.getElementById('canvas-spinner'),
  spinnerText: document.getElementById('spinner-text'),
  
  toolSelect: document.getElementById('tool-select'),
  toolDraw: document.getElementById('tool-draw'),
  btnZoomIn: document.getElementById('btn-zoom-in'),
  btnZoomOut: document.getElementById('btn-zoom-out'),
  btnZoomFit: document.getElementById('btn-zoom-fit'),
  zoomText: document.getElementById('zoom-level-text'),
  
  toggleYolo: document.getElementById('toggle-yolo'),
  toggleDino: document.getElementById('toggle-dino'),
  toggleSam: document.getElementById('toggle-sam'),
  
  btnPrevImg: document.getElementById('btn-prev-img'),
  btnNextImg: document.getElementById('btn-next-img'),
  paginationText: document.getElementById('image-pagination-text'),
  btnAcceptVerify: document.getElementById('btn-accept-verify'),
  btnAccept: document.getElementById('btn-accept'),
  btnReject: document.getElementById('btn-reject'),
  btnNextDifficult: document.getElementById('btn-next-difficult'),
  
  // Provenance Elements
  provPipeline: document.getElementById('prov-pipeline'),
  provQwen: document.getElementById('prov-qwen'),
  provHumanStatus: document.getElementById('prov-human-status'),
  provIouRow: document.getElementById('prov-iou-row'),
  provIouVal: document.getElementById('prov-iou-val'),
  provPriorityRow: document.getElementById('prov-priority-row'),
  provPriorityBadge: document.getElementById('prov-priority-badge'),
  
  // Active Learning Queue Elements
  alCountBadge: document.getElementById('al-count-badge'),
  alFilterSelect: document.getElementById('al-filter-select'),
  btnAlReviewNext: document.getElementById('btn-al-review-next'),
  btnAlExportHard: document.getElementById('btn-al-export-hard'),
  alQueueList: document.getElementById('al-queue-list'),
  
  // Test Set & Coverage Elements
  btnToggleTestSet: document.getElementById('btn-toggle-test-set'),
  btnTestSetText: document.getElementById('btn-test-set-text'),
  testSetPill: document.getElementById('test-set-pill'),
  btnRefreshCoverage: document.getElementById('btn-refresh-coverage'),
  btnExportTestSet: document.getElementById('btn-export-test-set'),
  btnScanDuplicates: document.getElementById('btn-scan-duplicates'),
  coverageWarningsList: document.getElementById('coverage-warnings-list'),
  covPositives: document.getElementById('cov-positives'),
  covHardPos: document.getElementById('cov-hard-pos'),
  covNegatives: document.getElementById('cov-negatives'),
  covHardNeg: document.getElementById('cov-hard-neg'),
  covFalsePos: document.getElementById('cov-false-pos'),
  covMissed: document.getElementById('cov-missed'),
  covTestSet: document.getElementById('cov-test-set'),
  covDuplicates: document.getElementById('cov-duplicates'),

  // Reject Modal Elements
  rejectModal: document.getElementById('reject-modal'),
  btnCloseRejectModal: document.getElementById('btn-close-reject-modal'),
  btnCancelReject: document.getElementById('btn-cancel-reject'),
  btnConfirmReject: document.getElementById('btn-confirm-reject'),
  rejectFailureType: document.getElementById('reject-failure-type'),
  rejectNegativeCategory: document.getElementById('reject-negative-category'),
  
  // Right Sidebar Inspector
  selectionStatusBadge: document.getElementById('selection-status-badge'),
  selectionEmptyState: document.getElementById('selection-empty-state'),
  selectionDetails: document.getElementById('selection-details'),
  boxClassSelect: document.getElementById('box-class-select'),
  boxSourceBadge: document.getElementById('box-source-badge'),
  confidenceBar: document.getElementById('confidence-bar'),
  boxConfVal: document.getElementById('box-conf-val'),
  boxReviewBanner: document.getElementById('box-review-banner'),
  boxReviewReasons: document.getElementById('box-review-reasons'),
  attrTilt: document.getElementById('attr-tilt'),
  attrArea: document.getElementById('attr-area'),
  attrC0: document.getElementById('attr-c0'),
  btnInspectRefineSam: document.getElementById('btn-inspect-refine-sam'),
  btnInspectDelete: document.getElementById('btn-inspect-delete'),
  
  diffStats: document.getElementById('diff-stats'),
  diffHumanCount: document.getElementById('diff-human-count'),
  diffModifiedCount: document.getElementById('diff-modified-count'),
  diffDeletedCount: document.getElementById('diff-deleted-count'),
  modelStatusList: document.getElementById('model-status-list'),
  
  // Modals
  batchModal: document.getElementById('batch-modal'),
  btnCloseBatchModal: document.getElementById('btn-close-batch-modal'),
  batchModeSelect: document.getElementById('batch-mode-select'),
  batchUnlabeledOnly: document.getElementById('batch-unlabeled-only'),
  batchSamRefine: document.getElementById('batch-sam-refine'),
  batchProgressFill: document.getElementById('batch-progress-fill'),
  batchStatusVal: document.getElementById('batch-status-val'),
  batchProgressVal: document.getElementById('batch-progress-val'),
  batchAcceptedVal: document.getElementById('batch-accepted-val'),
  batchReviewVal: document.getElementById('batch-review-val'),
  batchFailedVal: document.getElementById('batch-failed-val'),
  btnStartBatch: document.getElementById('btn-start-batch'),
  btnPauseBatch: document.getElementById('btn-pause-batch'),
  btnResumeBatch: document.getElementById('btn-resume-batch'),
  btnCancelBatch: document.getElementById('btn-cancel-batch'),
  
  importModal: document.getElementById('import-modal'),
  btnCloseImportModal: document.getElementById('btn-close-import-modal'),
  importSourcePath: document.getElementById('import-source-path'),
  btnRunImport: document.getElementById('btn-run-import'),
  btnCancelImport: document.getElementById('btn-cancel-import'),
  
  shortcutsModal: document.getElementById('shortcuts-modal'),
  btnCloseShortcutsModal: document.getElementById('btn-close-shortcuts-modal'),
};

const ctx = elements.canvas.getContext('2d');

// --- GEOMETRY UTILITIES (CLIENT-SIDE) ---

function signedShoelaceArea(pts) {
  if (!pts || pts.length < 3) return 0;
  let area2 = 0;
  for (let i = 0; i < pts.length; i++) {
    const p1 = pts[i];
    const p2 = pts[(i + 1) % pts.length];
    area2 += (p1[0] * p2[1] - p2[0] * p1[1]);
  }
  return area2 / 2.0;
}

function orderCornersCanonical(pts) {
  if (!pts || pts.length !== 4) return pts;
  // 1. Centroid
  let cx = 0, cy = 0;
  for (const p of pts) { cx += p[0]; cy += p[1]; }
  cx /= 4; cy /= 4;
  
  // 2. Sort by angle in screen coords
  const sorted = [...pts].sort((a, b) => {
    const angA = Math.atan2(a[1] - cy, a[0] - cx);
    const angB = Math.atan2(b[1] - cy, b[0] - cx);
    return angA - angB;
  });
  
  // 3. Verify clockwise winding (area > 0)
  if (signedShoelaceArea(sorted) < 0) {
    sorted.reverse();
  }
  
  // 4. Find Top-Left (min x + y)
  let bestIdx = 0;
  let minScore = Infinity;
  for (let i = 0; i < 4; i++) {
    const score = sorted[i][0] + sorted[i][1];
    if (score < minScore - 1e-5) {
      minScore = score;
      bestIdx = i;
    }
  }
  
  // 5. Roll array so bestIdx is 0
  const out = [];
  for (let i = 0; i < 4; i++) {
    out.push(sorted[(bestIdx + i) % 4]);
  }
  return out;
}

function computeTiltAngle(corners) {
  if (!corners || corners.length < 4) return 0.0;
  // Vector from bottom center to top center
  const topCx = (corners[0][0] + corners[1][0]) / 2;
  const topCy = (corners[0][1] + corners[1][1]) / 2;
  const botCx = (corners[3][0] + corners[2][0]) / 2;
  const botCy = (corners[3][1] + corners[2][1]) / 2;
  
  const dx = topCx - botCx;
  const dy = topCy - botCy;
  // Vertical axis points upwards in math (-dy in screen coords)
  const angleRad = Math.atan2(dx, -dy);
  return (angleRad * 180 / Math.PI);
}

// --- INITIALIZATION ---

async function init() {
  setupEventListeners();
  resizeCanvas();
  window.addEventListener('resize', () => { resizeCanvas(); render(); });
  
  await fetchHardware();
  await fetchModelStatus();
  await loadDatasets();
}

async function fetchHardware() {
  try {
    const res = await fetch('/api/system/hardware');
    const data = await res.json();
    state.activeDevice = data.active_device_setting;
    elements.deviceSelect.value = state.activeDevice;
    elements.hardwareLabel.textContent = `${data.resolved_device.toUpperCase()} ${data.hardware.gpu_name ? '(' + data.hardware.gpu_name + ')' : ''}`;
  } catch (err) {
    console.error('Failed to fetch hardware:', err);
  }
}

async function fetchModelStatus() {
  try {
    const res = await fetch('/api/models/status');
    const data = await res.json();
    elements.modelStatusList.innerHTML = '';
    data.models.forEach(m => {
      const row = document.createElement('div');
      row.className = 'model-row';
      
      let badgeClass = 'standby';
      let labelText = 'standby';
      if (m.status === 'ready') {
        badgeClass = 'ready';
        labelText = 'ready';
      } else if (m.status === 'not_loaded') {
        badgeClass = 'standby';
        labelText = 'standby (click to load)';
      } else if (m.status === 'missing_weights') {
        badgeClass = 'missing_weights';
        labelText = 'no weights';
      } else if (m.status === 'unavailable') {
        badgeClass = 'unavailable';
        labelText = 'unavailable';
      }
      
      row.innerHTML = `
        <span class="m-name">${m.name}</span>
        <span class="m-badge ${badgeClass}" style="cursor: pointer;" title="Status: ${m.status}. Click to pre-load into memory.">${labelText}</span>
      `;
      
      // Allow user to click badge to pre-load model
      const badge = row.querySelector('.m-badge');
      if (m.status === 'not_loaded') {
        badge.addEventListener('click', async () => {
          badge.textContent = 'loading...';
          try {
            await fetch(`/api/models/load?model_id=${encodeURIComponent(m.id)}`, { method: 'POST' });
            await fetchModelStatus();
          } catch (e) {
            badge.textContent = 'error';
          }
        });
      }
      
      elements.modelStatusList.appendChild(row);
    });
  } catch (err) {
    console.error('Failed to fetch model status:', err);
  }
}

async function loadDatasets() {
  try {
    const res = await fetch('/api/datasets');
    const data = await res.json();
    state.datasets = data.datasets || [];
    
    elements.datasetSelect.innerHTML = '';
    if (state.datasets.length === 0) {
      // Create initial dataset
      await createNewDataset('utility_poles_v1', 'Utility Poles Primary');
      return;
    }
    
    state.datasets.forEach(ds => {
      const opt = document.createElement('option');
      opt.value = ds.dataset_id;
      opt.textContent = `${ds.name} (${ds.image_count})`;
      elements.datasetSelect.appendChild(opt);
    });
    
    state.activeDatasetId = state.datasets[0].dataset_id;
    elements.datasetSelect.value = state.activeDatasetId;
    await loadDatasetImages(state.activeDatasetId);
  } catch (err) {
    console.error('Failed to load datasets:', err);
  }
}

async function createNewDataset(id, name) {
  try {
    const res = await fetch('/api/datasets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset_id: id, name: name, classes: ['utility_pole'] })
    });
    const data = await res.json();
    await loadDatasets();
  } catch (err) {
    alert('Failed to create dataset: ' + err.message);
  }
}

async function loadDatasetImages(datasetId) {
  try {
    const filterParam = state.filterStatus === 'all' ? '' : `?status=${state.filterStatus}`;
    const res = await fetch(`/api/datasets/${datasetId}/images${filterParam}`);
    const data = await res.json();
    state.images = data.images || [];
    
    // Update count badge & stats
    const dsMetaRes = await fetch(`/api/datasets/${datasetId}`);
    const dsData = await dsMetaRes.json();
    const meta = dsData.dataset;
    elements.imageCountBadge.textContent = meta.image_count;
    elements.statVerified.textContent = meta.verified_count;
    elements.statReview.textContent = meta.needs_review_count;
    elements.statUnlabeled.textContent = meta.unlabeled_count;
    
    renderGallery();
    
    // Load active learning queue and dataset composition alongside gallery
    loadActiveLearningQueue();
    loadDatasetComposition();
    
    if (state.images.length > 0) {
      selectImage(0);
    } else {
      state.activeImageIndex = -1;
      state.activeImageMeta = null;
      state.boxes = [];
      state.imageObj = null;
      elements.paginationText.textContent = 'No images in dataset';
      render();
    }
  } catch (err) {
    console.error('Failed to load images:', err);
  }
}

function renderGallery() {
  elements.imageGallery.innerHTML = '';
  state.images.forEach((img, idx) => {
    const item = document.createElement('div');
    item.className = `gallery-item ${idx === state.activeImageIndex ? 'active' : ''}`;
    
    let statusClass = 'unlabeled';
    if (img.status === 'verified' || img.status === 'accepted') statusClass = 'verified';
    else if (img.status === 'human_corrected') statusClass = 'human_corrected';
    else if (img.needs_review) statusClass = 'review';
    else if (img.status === 'ai_suggested') statusClass = 'ai_suggested';
    else if (img.status === 'rejected') statusClass = 'rejected';

    let extraBadges = '';
    if (img.is_test_set) extraBadges += ' <span class="badge-mini test">TEST</span>';
    if (img.is_duplicate) extraBadges += ' <span class="badge-mini dup">DUP</span>';
    
    item.innerHTML = `
      <div class="gallery-item-left">
        <span class="gallery-filename" title="${img.filename}">${img.filename}</span>
        ${extraBadges}
      </div>
      <span class="status-tag ${statusClass}">${img.status.replace('_', ' ')}</span>
    `;
    item.addEventListener('click', () => selectImage(idx));
    elements.imageGallery.appendChild(item);
  });
}

async function selectImage(index) {
  if (index < 0 || index >= state.images.length) return;
  state.activeImageIndex = index;
  state.activeImageMeta = state.images[index];
  state.selectedBoxIndex = -1;
  updateSelectionInspector();
  updateTestSetUI(!!state.activeImageMeta.is_test_set);
  setPositiveTags(state.activeImageMeta.al_categories || []);
  renderGallery();
  
  elements.paginationText.textContent = `${index + 1} / ${state.images.length} — ${state.activeImageMeta.filename}`;
  
  showSpinner('Loading image & labels...');
  
  // 1. Load image onto HTML Image element
  const imgUrl = `/api/datasets/${state.activeDatasetId}/images/${encodeURIComponent(state.activeImageMeta.filename)}`;
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.src = imgUrl;
  await new Promise((resolve, reject) => {
    img.onload = () => { state.imageObj = img; resolve(); };
    img.onerror = () => { console.error('Error loading image'); resolve(); };
  });
  
  // 2. Fetch existing annotations or predictions
  try {
    const annRes = await fetch(`/api/datasets/${state.activeDatasetId}/images/${encodeURIComponent(state.activeImageMeta.filename)}/annotations`);
    const annData = await annRes.json();
    
    state.boxes = [];
    if (annData.data && annData.data.boxes) {
      state.boxes = annData.data.boxes.map(b => {
        let corners = b.corners;
        if (!corners && b.xyxy) {
          const [x1, y1, x2, y2] = b.xyxy;
          corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]];
        }
        return {
          ...b,
          corners: orderCornersCanonical(corners)
        };
      });
    }
  } catch (err) {
    console.error('Failed to load annotations:', err);
  }
  
  // 3. Fetch differential history for this image
  await fetchDifferentialHistory(state.activeImageMeta.filename);
  
  hideSpinner();
  zoomFit();
}

async function fetchDifferentialHistory(filename) {
  try {
    const res = await fetch(`/api/datasets/${state.activeDatasetId}/history?filename=${encodeURIComponent(filename)}`);
    const data = await res.json();
    if (data.history && data.history.length > 0) {
      const h = data.history[0];
      elements.diffStats.classList.remove('hidden');
      elements.diffHumanCount.textContent = h.human_box_count || 0;
      elements.diffModifiedCount.textContent = h.modified_count || 0;
      elements.diffDeletedCount.textContent = h.deleted_count || 0;
    } else {
      elements.diffStats.classList.add('hidden');
    }
  } catch (err) {
    console.error('Failed to fetch history:', err);
  }
}

// --- CANVAS RENDERING ENGINE ---

function resizeCanvas() {
  const rect = elements.viewport.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  elements.canvas.width = rect.width * dpr;
  elements.canvas.height = rect.height * dpr;
  elements.canvas.style.width = `${rect.width}px`;
  elements.canvas.style.height = `${rect.height}px`;
  ctx.scale(dpr, dpr);
}

function zoomFit() {
  if (!state.imageObj) return;
  const rect = elements.viewport.getBoundingClientRect();
  const imgW = state.imageObj.width;
  const imgH = state.imageObj.height;
  
  const scaleX = (rect.width - 60) / imgW;
  const scaleY = (rect.height - 60) / imgH;
  const scale = Math.min(scaleX, scaleY, 1.5);
  
  const tx = (rect.width - imgW * scale) / 2;
  const ty = (rect.height - imgH * scale) / 2;
  
  state.transform = { scale, tx, ty };
  elements.zoomText.textContent = `${Math.round(scale * 100)}%`;
  render();
}

function render() {
  const rect = elements.viewport.getBoundingClientRect();
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
  ctx.restore();
  
  if (!state.imageObj) return;
  
  ctx.save();
  // Apply Zoom & Pan
  ctx.translate(state.transform.tx, state.transform.ty);
  ctx.scale(state.transform.scale, state.transform.scale);
  
  // 1. Draw Image
  ctx.drawImage(state.imageObj, 0, 0);
  
  // 2. Draw Active Boxes
  state.boxes.forEach((box, idx) => {
    // Check layer visibility filters
    const src = (box.model_source || '').toLowerCase();
    if (src.includes('yolo') && !state.layerVisibility.yolo && !src.includes('human')) return;
    if (src.includes('dino') && !state.layerVisibility.dino && !src.includes('human')) return;
    if (src.includes('sam') && !state.layerVisibility.sam && !src.includes('human')) return;
    
    const isSelected = idx === state.selectedBoxIndex;
    const isHovered = idx === state.hoveredBoxIndex;
    drawOBB(box, isSelected, isHovered);
  });
  
  // 3. Draw In-Progress Drawing Box
  if (state.isDrawing && state.drawStart && state.currentMousePos) {
    drawInteractiveRect(state.drawStart, state.currentMousePos);
  }
  
  ctx.restore();
}

function drawOBB(box, isSelected, isHovered) {
  const corners = box.corners;
  if (!corners || corners.length !== 4) return;
  
  // Determine Color Scheme based on Review / Source / Confidence
  let strokeColor = '#58a6ff'; // Blue for human
  let fillColor = 'rgba(88, 166, 255, 0.2)';
  
  if (box.model_source !== 'HUMAN') {
    if (box.needs_review || box.confidence < 0.45) {
      strokeColor = '#f85149'; // Red (review / low conf)
      fillColor = 'rgba(248, 81, 73, 0.2)';
    } else if (box.confidence < 0.70 || box.model_source === 'DINO') {
      strokeColor = '#d29922'; // Yellow (medium)
      fillColor = 'rgba(210, 153, 34, 0.2)';
    } else {
      strokeColor = '#2ea043'; // Green (high conf)
      fillColor = 'rgba(46, 160, 67, 0.2)';
    }
  }
  
  if (isSelected) {
    strokeColor = '#39c5bb'; // Cyan highlight for selected
    fillColor = 'rgba(57, 197, 187, 0.25)';
  }
  
  // Draw Polygon Path (Clockwise: 0 -> 1 -> 2 -> 3 -> 0)
  ctx.beginPath();
  ctx.moveTo(corners[0][0], corners[0][1]);
  for (let i = 1; i < 4; i++) {
    ctx.lineTo(corners[i][0], corners[i][1]);
  }
  ctx.closePath();
  ctx.fillStyle = fillColor;
  ctx.fill();
  ctx.lineWidth = isSelected ? 2.5 / state.transform.scale : 1.5 / state.transform.scale;
  ctx.strokeStyle = strokeColor;
  ctx.stroke();
  
  // Draw Centerline / Orientation Vector (Pole Axis)
  const topCx = (corners[0][0] + corners[1][0]) / 2;
  const topCy = (corners[0][1] + corners[1][1]) / 2;
  const botCx = (corners[3][0] + corners[2][0]) / 2;
  const botCy = (corners[3][1] + corners[2][1]) / 2;
  
  ctx.beginPath();
  ctx.setLineDash([4 / state.transform.scale, 4 / state.transform.scale]);
  ctx.moveTo(botCx, botCy);
  ctx.lineTo(topCx, topCy);
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
  ctx.stroke();
  ctx.setLineDash([]);
  
  // Draw Handles if Selected or Hovered
  if (isSelected || isHovered) {
    const handleRadius = 5 / state.transform.scale;
    
    // Corner 0: Solid Canonical Top-Left Anchor
    ctx.beginPath();
    ctx.arc(corners[0][0], corners[0][1], handleRadius * 1.3, 0, Math.PI * 2);
    ctx.fillStyle = '#ff7b72'; // Red-orange anchor
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5 / state.transform.scale;
    ctx.stroke();
    
    // Corners 1, 2, 3: Hollow Handles
    for (let i = 1; i < 4; i++) {
      ctx.beginPath();
      ctx.arc(corners[i][0], corners[i][1], handleRadius, 0, Math.PI * 2);
      ctx.fillStyle = '#fff';
      ctx.fill();
      ctx.strokeStyle = strokeColor;
      ctx.stroke();
    }
    
    // Rotation Handle (Antenna extended above top edge)
    const antennaLen = 22 / state.transform.scale;
    const dx = topCx - botCx;
    const dy = topCy - botCy;
    const len = Math.hypot(dx, dy) || 1;
    const rotX = topCx + (dx / len) * antennaLen;
    const rotY = topCy + (dy / len) * antennaLen;
    
    ctx.beginPath();
    ctx.moveTo(topCx, topCy);
    ctx.lineTo(rotX, rotY);
    ctx.strokeStyle = '#39c5bb';
    ctx.stroke();
    
    ctx.beginPath();
    ctx.arc(rotX, rotY, handleRadius * 1.1, 0, Math.PI * 2);
    ctx.fillStyle = '#39c5bb';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.stroke();
  }
  
  // Draw Label Badge
  const badgeX = corners[0][0];
  const badgeY = corners[0][1] - 8 / state.transform.scale;
  ctx.font = `${Math.max(10, 11 / state.transform.scale)}px ${getComputedStyle(document.body).fontFamily}`;
  const confText = box.confidence ? `${Math.round(box.confidence * 100)}%` : '';
  const labelStr = `${box.class_name || 'utility_pole'} [${box.model_source || 'AI'}] ${confText}`;
  const textW = ctx.measureText(labelStr).width;
  
  ctx.fillStyle = 'rgba(18, 23, 31, 0.85)';
  ctx.fillRect(badgeX, badgeY - 14 / state.transform.scale, textW + 8 / state.transform.scale, 16 / state.transform.scale);
  ctx.fillStyle = '#f0f6fc';
  ctx.fillText(labelStr, badgeX + 4 / state.transform.scale, badgeY - 2 / state.transform.scale);
}

function drawInteractiveRect(start, current) {
  const minX = Math.min(start.x, current.x);
  const maxX = Math.max(start.x, current.x);
  const minY = Math.min(start.y, current.y);
  const maxY = Math.max(start.y, current.y);
  
  ctx.beginPath();
  ctx.rect(minX, minY, maxX - minX, maxY - minY);
  ctx.strokeStyle = '#39c5bb';
  ctx.lineWidth = 1.5 / state.transform.scale;
  ctx.fillStyle = 'rgba(57, 197, 187, 0.15)';
  ctx.fill();
  ctx.stroke();
}

// --- MOUSE & CANVAS INTERACTION ---

function getCanvasCoords(evt) {
  const rect = elements.canvas.getBoundingClientRect();
  const clientX = evt.clientX;
  const clientY = evt.clientY;
  const canvasX = (clientX - rect.left - state.transform.tx) / state.transform.scale;
  const canvasY = (clientY - rect.top - state.transform.ty) / state.transform.scale;
  return { x: canvasX, y: canvasY, rawX: clientX, rawY: clientY };
}

function hitTest(x, y) {
  const tol = 8 / state.transform.scale;
  
  // 1. Check active selected box handles first
  if (state.selectedBoxIndex >= 0 && state.selectedBoxIndex < state.boxes.length) {
    const box = state.boxes[state.selectedBoxIndex];
    const c = box.corners;
    if (c && c.length === 4) {
      // Rotation handle
      const topCx = (c[0][0] + c[1][0]) / 2;
      const topCy = (c[0][1] + c[1][1]) / 2;
      const botCx = (c[3][0] + c[2][0]) / 2;
      const botCy = (c[3][1] + c[2][1]) / 2;
      const dx = topCx - botCx, dy = topCy - botCy;
      const len = Math.hypot(dx, dy) || 1;
      const antennaLen = 22 / state.transform.scale;
      const rotX = topCx + (dx / len) * antennaLen;
      const rotY = topCy + (dy / len) * antennaLen;
      
      if (Math.hypot(x - rotX, y - rotY) <= tol) {
        return { boxIdx: state.selectedBoxIndex, handle: 'rotate' };
      }
      // Corners 0, 1, 2, 3
      for (let i = 0; i < 4; i++) {
        if (Math.hypot(x - c[i][0], y - c[i][1]) <= tol) {
          return { boxIdx: state.selectedBoxIndex, handle: `c${i}` };
        }
      }
    }
  }
  
  // 2. Check inside polygon for all boxes (topmost first)
  for (let b = state.boxes.length - 1; b >= 0; b--) {
    const c = state.boxes[b].corners;
    if (c && isPointInPolygon(x, y, c)) {
      return { boxIdx: b, handle: 'body' };
    }
  }
  
  return { boxIdx: -1, handle: null };
}

function isPointInPolygon(x, y, vs) {
  let inside = false;
  for (let i = 0, j = vs.length - 1; i < vs.length; j = i++) {
    const xi = vs[i][0], yi = vs[i][1];
    const xj = vs[j][0], yj = vs[j][1];
    const intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

// --- EVENT LISTENERS ---

function setupEventListeners() {
  // Zoom & Pan
  elements.viewport.addEventListener('wheel', handleWheel, { passive: false });
  elements.viewport.addEventListener('mousedown', handleMouseDown);
  window.addEventListener('mousemove', handleMouseMove);
  window.addEventListener('mouseup', handleMouseUp);
  
  // Toolbar buttons
  elements.btnZoomIn.addEventListener('click', () => zoomBy(1.2));
  elements.btnZoomOut.addEventListener('click', () => zoomBy(0.8));
  elements.btnZoomFit.addEventListener('click', zoomFit);
  
  elements.toolSelect.addEventListener('click', () => setToolMode('select'));
  elements.toolDraw.addEventListener('click', () => setToolMode('draw'));
  
  // Layer Toggles
  elements.toggleYolo.addEventListener('change', (e) => { state.layerVisibility.yolo = e.target.checked; render(); });
  elements.toggleDino.addEventListener('change', (e) => { state.layerVisibility.dino = e.target.checked; render(); });
  elements.toggleSam.addEventListener('change', (e) => { state.layerVisibility.sam = e.target.checked; render(); });
  
  // Top nav action buttons
  elements.btnAiLabel.addEventListener('click', () => triggerInference('AI_LABEL'));
  elements.btnRunAll.addEventListener('click', () => triggerInference('RUN_ALL'));
  elements.btnRefineSam.addEventListener('click', refineSelectedWithSAM);
  elements.btnExportDataset.addEventListener('click', exportDataset);
  
  // Navigation & Save
  elements.btnPrevImg.addEventListener('click', () => selectImage(state.activeImageIndex - 1));
  elements.btnNextImg.addEventListener('click', () => selectImage(state.activeImageIndex + 1));
  elements.btnAcceptVerify.addEventListener('click', saveEditsCurrentAnnotation);
  
  // Accept / Reject / Next Difficult
  elements.btnAccept.addEventListener('click', acceptCurrentAnnotation);
  elements.btnReject.addEventListener('click', rejectCurrentAnnotation);
  elements.btnNextDifficult.addEventListener('click', loadNextDifficultImage);
  
  // Active Learning Controls
  elements.btnAlReviewNext.addEventListener('click', loadNextDifficultImage);
  elements.btnAlExportHard.addEventListener('click', exportHardCases);
  elements.alFilterSelect.addEventListener('change', () => loadActiveLearningQueue());
  
  // Test Set & Coverage Controls
  if (elements.btnToggleTestSet) elements.btnToggleTestSet.addEventListener('click', toggleTestSetCurrent);
  if (elements.btnRefreshCoverage) elements.btnRefreshCoverage.addEventListener('click', loadDatasetComposition);
  if (elements.btnExportTestSet) elements.btnExportTestSet.addEventListener('click', exportTestSet);
  if (elements.btnScanDuplicates) elements.btnScanDuplicates.addEventListener('click', scanDuplicates);

  // Detailed Reject Modal Controls
  if (elements.btnCloseRejectModal) elements.btnCloseRejectModal.addEventListener('click', () => elements.rejectModal.classList.add('hidden'));
  if (elements.btnCancelReject) elements.btnCancelReject.addEventListener('click', () => elements.rejectModal.classList.add('hidden'));
  if (elements.btnConfirmReject) elements.btnConfirmReject.addEventListener('click', confirmRejectCurrent);
  
  // Inspector
  elements.btnInspectRefineSam.addEventListener('click', refineSelectedWithSAM);
  elements.btnInspectDelete.addEventListener('click', deleteSelectedBox);
  
  // Filter pills
  document.querySelectorAll('.filter-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      document.querySelectorAll('.filter-pill').forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      state.filterStatus = pill.getAttribute('data-filter');
      loadDatasetImages(state.activeDatasetId);
    });
  });
  
  // Dataset change
  elements.datasetSelect.addEventListener('change', (e) => {
    state.activeDatasetId = e.target.value;
    loadDatasetImages(state.activeDatasetId);
  });
  
  // Device change
  elements.deviceSelect.addEventListener('change', async (e) => {
    await fetch('/api/system/device', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ device: e.target.value })
    });
    await fetchHardware();
  });
  
  // Modals
  elements.btnBatchModal.addEventListener('click', () => elements.batchModal.classList.remove('hidden'));
  elements.btnCloseBatchModal.addEventListener('click', () => elements.batchModal.classList.add('hidden'));
  elements.btnStartBatch.addEventListener('click', startBatchJob);
  elements.btnPauseBatch.addEventListener('click', () => fetch('/api/batch/pause', { method: 'POST' }));
  elements.btnResumeBatch.addEventListener('click', () => fetch('/api/batch/resume', { method: 'POST' }));
  elements.btnCancelBatch.addEventListener('click', () => fetch('/api/batch/cancel', { method: 'POST' }));
  
  elements.btnImportImages.addEventListener('click', () => elements.importModal.classList.remove('hidden'));
  elements.btnCloseImportModal.addEventListener('click', () => elements.importModal.classList.add('hidden'));
  elements.btnCancelImport.addEventListener('click', () => elements.importModal.classList.add('hidden'));
  elements.btnRunImport.addEventListener('click', runImageImport);
  
  elements.btnShortcuts.addEventListener('click', () => elements.shortcutsModal.classList.toggle('hidden'));
  elements.btnCloseShortcutsModal.addEventListener('click', () => elements.shortcutsModal.classList.add('hidden'));
  
  // Keyboard Shortcuts
  window.addEventListener('keydown', handleKeyDown);
}

function setToolMode(mode) {
  state.toolMode = mode;
  elements.toolSelect.classList.toggle('active', mode === 'select');
  elements.toolDraw.classList.toggle('active', mode === 'draw');
}

function zoomBy(factor) {
  const rect = elements.viewport.getBoundingClientRect();
  const cx = rect.width / 2;
  const cy = rect.height / 2;
  const newScale = Math.min(Math.max(0.1, state.transform.scale * factor), 10.0);
  
  state.transform.tx = cx - (cx - state.transform.tx) * (newScale / state.transform.scale);
  state.transform.ty = cy - (cy - state.transform.ty) * (newScale / state.transform.scale);
  state.transform.scale = newScale;
  elements.zoomText.textContent = `${Math.round(newScale * 100)}%`;
  render();
}

function handleWheel(e) {
  e.preventDefault();
  const rect = elements.viewport.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;
  
  const factor = e.deltaY < 0 ? 1.15 : 0.85;
  const newScale = Math.min(Math.max(0.1, state.transform.scale * factor), 10.0);
  
  state.transform.tx = mouseX - (mouseX - state.transform.tx) * (newScale / state.transform.scale);
  state.transform.ty = mouseY - (mouseY - state.transform.ty) * (newScale / state.transform.scale);
  state.transform.scale = newScale;
  elements.zoomText.textContent = `${Math.round(newScale * 100)}%`;
  render();
}

function handleMouseDown(e) {
  const coords = getCanvasCoords(e);
  
  // Middle click or Spacebar held -> Pan
  if (e.button === 1 || e.spaceKey || e.altKey) {
    state.isPanning = true;
    state.panStart = { x: e.clientX - state.transform.tx, y: e.clientY - state.transform.ty };
    return;
  }
  
  if (e.button !== 0) return; // Left click only below
  
  if (state.toolMode === 'draw') {
    state.isDrawing = true;
    state.drawStart = { x: coords.x, y: coords.y };
    state.currentMousePos = { x: coords.x, y: coords.y };
    return;
  }
  
  // Select / Edit mode
  const hit = hitTest(coords.x, coords.y);
  if (hit.boxIdx >= 0) {
    state.selectedBoxIndex = hit.boxIdx;
    state.isDraggingHandle = true;
    state.dragTarget = {
      handle: hit.handle,
      startCoords: coords,
      origCorners: JSON.parse(JSON.stringify(state.boxes[hit.boxIdx].corners)),
      box: state.boxes[hit.boxIdx]
    };
    updateSelectionInspector();
    render();
  } else {
    // Clicked background
    state.selectedBoxIndex = -1;
    updateSelectionInspector();
    render();
  }
}

function handleMouseMove(e) {
  const coords = getCanvasCoords(e);
  state.currentMousePos = { x: coords.x, y: coords.y };
  
  if (state.isPanning) {
    state.transform.tx = e.clientX - state.panStart.x;
    state.transform.ty = e.clientY - state.panStart.y;
    render();
    return;
  }
  
  if (state.isDrawing) {
    render();
    return;
  }
  
  if (state.isDraggingHandle && state.dragTarget) {
    const handle = state.dragTarget.handle;
    const orig = state.dragTarget.origCorners;
    const dx = coords.x - state.dragTarget.startCoords.x;
    const dy = coords.y - state.dragTarget.startCoords.y;
    
    if (handle === 'body') {
      // Translate entire box
      for (let i = 0; i < 4; i++) {
        state.dragTarget.box.corners[i] = [orig[i][0] + dx, orig[i][1] + dy];
      }
    } else if (handle.startsWith('c')) {
      // Individual corner drag
      const cIdx = parseInt(handle[1]);
      state.dragTarget.box.corners[cIdx] = [orig[cIdx][0] + dx, orig[cIdx][1] + dy];
    } else if (handle === 'rotate') {
      // Rotation around box centroid
      let cx = 0, cy = 0;
      for (const p of orig) { cx += p[0]; cy += p[1]; }
      cx /= 4; cy /= 4;
      
      const startAngle = Math.atan2(state.dragTarget.startCoords.y - cy, state.dragTarget.startCoords.x - cx);
      const currAngle = Math.atan2(coords.y - cy, coords.x - cx);
      const delta = currAngle - startAngle;
      
      const cosD = Math.cos(delta);
      const sinD = Math.sin(delta);
      for (let i = 0; i < 4; i++) {
        const ox = orig[i][0] - cx;
        const oy = orig[i][1] - cy;
        state.dragTarget.box.corners[i] = [
          cx + ox * cosD - oy * sinD,
          cy + ox * sinD + oy * cosD
        ];
      }
    }
    
    // Mark box as human edited
    state.dragTarget.box.model_source = 'HUMAN';
    state.dragTarget.box.confidence = 1.0;
    state.dragTarget.box.needs_review = false;
    
    updateSelectionInspector();
    render();
    return;
  }
  
  // Hover cursor styling
  const hit = hitTest(coords.x, coords.y);
  state.hoveredBoxIndex = hit.boxIdx;
  state.hoveredHandle = hit.handle;
  
  if (hit.handle === 'rotate') {
    elements.viewport.style.cursor = 'grab';
  } else if (hit.handle && hit.handle.startsWith('c')) {
    elements.viewport.style.cursor = 'crosshair';
  } else if (hit.handle === 'body') {
    elements.viewport.style.cursor = 'move';
  } else {
    elements.viewport.style.cursor = state.toolMode === 'draw' ? 'crosshair' : 'default';
  }
}

function handleMouseUp(e) {
  if (state.isPanning) {
    state.isPanning = false;
  }
  
  if (state.isDrawing && state.drawStart) {
    const coords = getCanvasCoords(e);
    const minX = Math.min(state.drawStart.x, coords.x);
    const maxX = Math.max(state.drawStart.x, coords.x);
    const minY = Math.min(state.drawStart.y, coords.y);
    const maxY = Math.max(state.drawStart.y, coords.y);
    
    if (maxX - minX > 5 && maxY - minY > 5) {
      const canonicalCorners = orderCornersCanonical([
        [minX, minY],
        [maxX, minY],
        [maxX, maxY],
        [minX, maxY]
      ]);
      
      const newBox = {
        xyxy: [minX, minY, maxX, maxY],
        corners: canonicalCorners,
        confidence: 1.0,
        class_id: 0,
        class_name: 'utility_pole',
        model_source: 'HUMAN',
        needs_review: false,
        review_reasons: [],
        attributes: {}
      };
      state.boxes.push(newBox);
      state.selectedBoxIndex = state.boxes.length - 1;
      setToolMode('select');
      updateSelectionInspector();
    }
    state.isDrawing = false;
    state.drawStart = null;
    render();
  }
  
  if (state.isDraggingHandle && state.dragTarget) {
    // Canonicalize corners on drag release
    const box = state.dragTarget.box;
    box.corners = orderCornersCanonical(box.corners);
    state.isDraggingHandle = false;
    state.dragTarget = null;
    updateSelectionInspector();
    render();
  }
}

// --- INSPECTOR & ATTRIBUTES ---

function updateSelectionInspector() {
  if (state.selectedBoxIndex < 0 || state.selectedBoxIndex >= state.boxes.length) {
    elements.selectionEmptyState.classList.remove('hidden');
    elements.selectionDetails.classList.add('hidden');
    elements.selectionStatusBadge.textContent = 'None';
    elements.selectionStatusBadge.className = 'badge';
    return;
  }
  
  elements.selectionEmptyState.classList.add('hidden');
  elements.selectionDetails.classList.remove('hidden');
  
  const box = state.boxes[state.selectedBoxIndex];
  elements.selectionStatusBadge.textContent = `#${state.selectedBoxIndex + 1}`;
  
  // Source badge
  const src = box.model_source || 'AI';
  elements.boxSourceBadge.textContent = src;
  elements.boxSourceBadge.className = `source-badge ${src.toLowerCase().replace('+', '_')}`;
  
  // Confidence meter
  const confPercent = Math.round((box.confidence || 1.0) * 100);
  elements.confidenceBar.style.width = `${confPercent}%`;
  elements.boxConfVal.textContent = `${confPercent}%`;
  if (confPercent >= 70) elements.confidenceBar.style.background = 'var(--color-green)';
  else if (confPercent >= 45) elements.confidenceBar.style.background = 'var(--color-yellow)';
  else elements.confidenceBar.style.background = 'var(--color-red)';
  
  // Review Banner
  if (box.needs_review && box.review_reasons && box.review_reasons.length > 0) {
    elements.boxReviewBanner.classList.remove('hidden');
    elements.boxReviewReasons.innerHTML = box.review_reasons.map(r => `<li>${r}</li>`).join('');
  } else {
    elements.boxReviewBanner.classList.add('hidden');
  }
  
  // Attributes
  const tilt = computeTiltAngle(box.corners);
  elements.attrTilt.textContent = `${tilt.toFixed(1)}°`;
  const area = Math.abs(signedShoelaceArea(box.corners));
  elements.attrArea.textContent = `${Math.round(area)} px²`;
  if (box.corners && box.corners.length > 0) {
    elements.attrC0.textContent = `${Math.round(box.corners[0][0])}, ${Math.round(box.corners[0][1])}`;
  }
  
  // --- Provenance & Evidence Card ---
  // Pipeline source chain
  const srcLower = src.toLowerCase();
  let pipelineText = src;
  if (srcLower.includes('dino') && srcLower.includes('sam')) {
    pipelineText = 'Grounding DINO → SAM 2.1 → OBB';
  } else if (srcLower.includes('dino')) {
    pipelineText = 'Grounding DINO → OBB';
  } else if (srcLower.includes('sam')) {
    pipelineText = 'SAM 2.1 → OBB';
  } else if (srcLower.includes('yolo')) {
    pipelineText = 'YOLO-OBB';
  } else if (srcLower === 'human') {
    pipelineText = 'Human Annotation';
  }
  elements.provPipeline.textContent = pipelineText;
  
  // Qwen verifier status (from box attributes if available)
  if (box.attributes && box.attributes.qwen_verdict) {
    elements.provQwen.textContent = box.attributes.qwen_verdict;
    elements.provQwen.className = 'prov-v green';
  } else {
    elements.provQwen.textContent = 'UNAVAILABLE';
    elements.provQwen.className = 'prov-v yellow';
  }
  
  // Human review status
  const imgStatus = state.activeImageMeta ? (state.activeImageMeta.status || 'unlabeled') : 'unlabeled';
  elements.provHumanStatus.textContent = imgStatus.replace('_', ' ');
  elements.provHumanStatus.className = `prov-v ${imgStatus === 'accepted' || imgStatus === 'verified' ? 'green' : imgStatus === 'rejected' ? 'red' : 'yellow'}`;
  
  // Correction IoU (from box attributes if available)
  if (box.attributes && box.attributes.correction_iou !== undefined) {
    elements.provIouRow.classList.remove('hidden');
    const iou = (box.attributes.correction_iou * 100).toFixed(1);
    elements.provIouVal.textContent = `${iou}%`;
  } else {
    elements.provIouRow.style.display = '';
    elements.provIouVal.textContent = '--';
  }
  
  // Active Learning priority (from image meta)
  if (state.activeImageMeta && state.activeImageMeta.priority !== undefined) {
    const priority = state.activeImageMeta.priority;
    elements.provPriorityRow.style.display = '';
    if (priority >= 70) {
      elements.provPriorityBadge.textContent = `HIGH (${priority})`;
      elements.provPriorityBadge.className = 'badge badge-danger';
    } else if (priority >= 40) {
      elements.provPriorityBadge.textContent = `MEDIUM (${priority})`;
      elements.provPriorityBadge.className = 'badge badge-warning';
    } else {
      elements.provPriorityBadge.textContent = `NORMAL (${priority})`;
      elements.provPriorityBadge.className = 'badge';
    }
  } else {
    elements.provPriorityBadge.textContent = 'NORMAL';
    elements.provPriorityBadge.className = 'badge';
  }
}

function deleteSelectedBox() {
  if (state.selectedBoxIndex < 0 || state.selectedBoxIndex >= state.boxes.length) return;
  state.boxes.splice(state.selectedBoxIndex, 1);
  state.selectedBoxIndex = -1;
  updateSelectionInspector();
  render();
}

// --- AI INFERENCE INTEGRATION ---

async function triggerInference(mode) {
  if (!state.activeImageMeta) return;
  const msg = mode === 'AI_LABEL' 
    ? 'Running AI Label Waterfall (YOLO -> SAM)...' 
    : 'Running All Models (YOLO + Grounding DINO + SAM)... (~15s on CPU)';
  showSpinner(msg);
  try {
    const res = await fetch('/api/inference/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset_id: state.activeDatasetId,
        filename: state.activeImageMeta.filename,
        mode: mode,
        conf_threshold: 0.25,
        use_sam_refinement: true
      })
    });
    const data = await res.json();
    state.boxes = (data.boxes || []).map(b => ({
      ...b,
      corners: orderCornersCanonical(b.corners)
    }));
    state.selectedBoxIndex = state.boxes.length > 0 ? 0 : -1;
    updateSelectionInspector();
    render();
    await fetchModelStatus();
  } catch (err) {
    alert('Inference failed: ' + err.message);
  } finally {
    hideSpinner();
  }
}

async function refineSelectedWithSAM() {
  if (state.selectedBoxIndex < 0 || state.selectedBoxIndex >= state.boxes.length) {
    alert('Please select an OBB on the canvas to refine.');
    return;
  }
  const box = state.boxes[state.selectedBoxIndex];
  showSpinner('Refining with SAM segmentation...');
  try {
    let xyxy = box.xyxy;
    if (!xyxy && box.corners) {
      const xs = box.corners.map(p => p[0]);
      const ys = box.corners.map(p => p[1]);
      xyxy = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
    }
    
    const res = await fetch('/api/inference/segment_box', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset_id: state.activeDatasetId,
        filename: state.activeImageMeta.filename,
        box_xyxy: xyxy
      })
    });
    const data = await res.json();
    if (data.corners) {
      box.corners = orderCornersCanonical(data.corners);
      box.model_source = 'SAM';
      updateSelectionInspector();
      render();
    }
  } catch (err) {
    alert('SAM refinement failed: ' + err.message);
  } finally {
    hideSpinner();
  }
}

// --- ANNOTATION SAVE, REJECTION & ACTIVE LEARNING ---

function getSelectedPositiveTags() {
  const checked = document.querySelectorAll('input[name="al_tag"]:checked');
  return Array.from(checked).map(c => c.value);
}

function clearPositiveTags() {
  document.querySelectorAll('input[name="al_tag"]').forEach(c => c.checked = false);
}

function setPositiveTags(tags) {
  clearPositiveTags();
  if (!Array.isArray(tags)) return;
  tags.forEach(t => {
    const el = document.querySelector(`input[name="al_tag"][value="${t}"]`);
    if (el) el.checked = true;
  });
}

/**
 * Accept current AI predictions as ground truth without modifications.
 * Sends action='accepted' and positive AL categories to backend.
 */
async function acceptCurrentAnnotation() {
  if (!state.activeImageMeta || !state.imageObj) return;
  const tags = getSelectedPositiveTags();
  showSpinner('Accepting annotation as ground truth...');
  try {
    const res = await fetch(`/api/datasets/${state.activeDatasetId}/images/${encodeURIComponent(state.activeImageMeta.filename)}/annotations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_width: state.imageObj.width,
        image_height: state.imageObj.height,
        is_human: true,
        action: 'accepted',
        al_categories: tags.length > 0 ? tags : ['clear_positive'],
        boxes: state.boxes.map(b => ({
          ...b,
          needs_review: false
        }))
      })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    
    // Flash green feedback
    elements.btnAccept.classList.add('flash-success');
    setTimeout(() => elements.btnAccept.classList.remove('flash-success'), 600);
    
    // Refresh stats and advance to next image
    await loadDatasetImages(state.activeDatasetId);
    if (state.activeImageIndex < state.images.length - 1) {
      selectImage(state.activeImageIndex + 1);
    }
  } catch (err) {
    alert('Accept failed: ' + err.message);
  } finally {
    hideSpinner();
  }
}

/**
 * Save human corrections to the annotation (when corners were dragged,
 * boxes added/deleted, etc). Sends action='human_corrected' to backend.
 */
async function saveEditsCurrentAnnotation() {
  if (!state.activeImageMeta || !state.imageObj) return;
  const tags = getSelectedPositiveTags();
  showSpinner('Saving human corrections...');
  try {
    const res = await fetch(`/api/datasets/${state.activeDatasetId}/images/${encodeURIComponent(state.activeImageMeta.filename)}/annotations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_width: state.imageObj.width,
        image_height: state.imageObj.height,
        is_human: true,
        action: 'human_corrected',
        al_categories: tags,
        boxes: state.boxes.map(b => ({
          ...b,
          model_source: b.model_source || 'HUMAN',
          confidence: 1.0,
          needs_review: false
        }))
      })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    
    // Flash blue feedback
    elements.btnAcceptVerify.classList.add('flash-info');
    setTimeout(() => elements.btnAcceptVerify.classList.remove('flash-info'), 600);
    
    // Refresh stats and advance
    await loadDatasetImages(state.activeDatasetId);
    if (state.activeImageIndex < state.images.length - 1) {
      selectImage(state.activeImageIndex + 1);
    }
  } catch (err) {
    alert('Save edits failed: ' + err.message);
  } finally {
    hideSpinner();
  }
}

// Legacy alias kept for backward compatibility
const acceptAndVerifyCurrent = saveEditsCurrentAnnotation;

/**
 * Open detailed rejection modal.
 */
function rejectCurrentAnnotation() {
  if (!state.activeImageMeta) return;
  elements.rejectModal.classList.remove('hidden');
}

/**
 * Confirm rejection with structured failure type and negative object category.
 */
async function confirmRejectCurrent() {
  if (!state.activeImageMeta) return;
  const failureType = elements.rejectFailureType.value;
  const negativeCategory = elements.rejectNegativeCategory.value;
  elements.rejectModal.classList.add('hidden');
  
  showSpinner('Recording negative / candidate failure...');
  try {
    const res = await fetch(`/api/datasets/${state.activeDatasetId}/images/${encodeURIComponent(state.activeImageMeta.filename)}/reject`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        failure_type: failureType,
        negative_category: negativeCategory,
        al_categories: [failureType, negativeCategory]
      })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    
    elements.btnReject.classList.add('flash-danger');
    setTimeout(() => elements.btnReject.classList.remove('flash-danger'), 600);
    
    await loadDatasetImages(state.activeDatasetId);
    if (state.activeImageIndex < state.images.length - 1) {
      selectImage(state.activeImageIndex + 1);
    }
  } catch (err) {
    alert('Reject failed: ' + err.message);
  } finally {
    hideSpinner();
  }
}

/**
 * Toggle fixed test set membership for current image.
 */
async function toggleTestSetCurrent() {
  if (!state.activeImageMeta) return;
  showSpinner('Updating test set isolation...');
  try {
    const res = await fetch(`/api/datasets/${state.activeDatasetId}/images/${encodeURIComponent(state.activeImageMeta.filename)}/test_set`, {
      method: 'POST'
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    state.activeImageMeta.is_test_set = data.is_test_set;
    updateTestSetUI(data.is_test_set);
    loadDatasetComposition();
    renderGallery();
  } catch (err) {
    alert('Failed to toggle test set: ' + err.message);
  } finally {
    hideSpinner();
  }
}

function updateTestSetUI(isTestSet) {
  if (!elements.btnToggleTestSet) return;
  if (isTestSet) {
    elements.btnTestSetText.textContent = 'In Test Set';
    elements.testSetPill.classList.remove('hidden');
    elements.btnToggleTestSet.classList.add('btn-primary');
    elements.btnToggleTestSet.classList.remove('btn-outline');
  } else {
    elements.btnTestSetText.textContent = 'Test Set';
    elements.testSetPill.classList.add('hidden');
    elements.btnToggleTestSet.classList.remove('btn-primary');
    elements.btnToggleTestSet.classList.add('btn-outline');
  }
}

/**
 * Load dataset composition counts and real-time coverage warnings.
 */
async function loadDatasetComposition() {
  if (!state.activeDatasetId) return;
  try {
    const res = await fetch(`/api/datasets/${state.activeDatasetId}/composition`);
    if (!res.ok) return;
    const comp = await res.json();
    
    if (elements.covPositives) elements.covPositives.textContent = comp.positive_poles || 0;
    if (elements.covHardPos) elements.covHardPos.textContent = comp.hard_positives || 0;
    if (elements.covNegatives) elements.covNegatives.textContent = comp.confirmed_negatives || 0;
    if (elements.covHardNeg) elements.covHardNeg.textContent = comp.hard_negatives || 0;
    if (elements.covFalsePos) elements.covFalsePos.textContent = comp.false_positives || 0;
    if (elements.covMissed) elements.covMissed.textContent = comp.missed_poles || 0;
    if (elements.covTestSet) elements.covTestSet.textContent = comp.test_set_count || 0;
    if (elements.covDuplicates) elements.covDuplicates.textContent = comp.near_duplicates || 0;
    
    const listEl = elements.coverageWarningsList;
    if (listEl) {
      listEl.innerHTML = '';
      const warnings = comp.warnings || [];
      if (warnings.length === 0) {
        listEl.innerHTML = '<div class="cov-clean">&#10003; Dataset balance & coverage are healthy</div>';
      } else {
        warnings.forEach(w => {
          const item = document.createElement('div');
          item.className = 'coverage-warning-banner';
          item.innerHTML = `<span class="warn-icon">&#9888;</span><span class="warn-text">${w}</span>`;
          listEl.appendChild(item);
        });
      }
    }
  } catch (err) {
    console.error('Failed to load dataset composition:', err);
  }
}

/**
 * Export dedicated FIXED TEST set partition.
 */
async function exportTestSet() {
  if (!state.activeDatasetId) return;
  showSpinner('Exporting fixed test set benchmark...');
  try {
    const res = await fetch(`/api/datasets/${state.activeDatasetId}/export_test_set`, { method: 'POST' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    alert(`Fixed Test Set Export Complete!\n\nPath: ${data.result.export_path}\nImages Exported: ${data.result.count}`);
  } catch (err) {
    alert('Export test set failed: ' + err.message);
  } finally {
    hideSpinner();
  }
}

/**
 * Scan for perceptual near-duplicates using pHash.
 */
async function scanDuplicates() {
  if (!state.activeDatasetId) return;
  showSpinner('Scanning perceptual near-duplicates (pHash)...');
  try {
    const res = await fetch(`/api/datasets/${state.activeDatasetId}/duplicates`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const dups = data.duplicates || [];
    alert(`Duplicate Scan Complete!\n\nFound ${dups.length} near-duplicate scene pairs. Down-ranked in Active Learning.`);
    await loadDatasetImages(state.activeDatasetId);
    await loadActiveLearningQueue();
    await loadDatasetComposition();
  } catch (err) {
    alert('Duplicate scan failed: ' + err.message);
  } finally {
    hideSpinner();
  }
}


/**
 * Navigate to the next most difficult image from the active learning queue.
 * Calls GET /api/datasets/{dataset_id}/next_difficult
 */
async function loadNextDifficultImage() {
  if (!state.activeDatasetId) return;
  showSpinner('Finding next difficult image...');
  try {
    const res = await fetch(`/api/datasets/${state.activeDatasetId}/next_difficult`);
    if (res.status === 404) {
      hideSpinner();
      alert('No more difficult images pending review. All images have been reviewed!');
      return;
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const nextItem = data.next_item;
    
    // Find image index in current gallery
    const idx = state.images.findIndex(img => img.filename === nextItem.filename);
    if (idx >= 0) {
      selectImage(idx);
    } else {
      // Image may be filtered out; reload without filter and find it
      state.filterStatus = 'all';
      document.querySelectorAll('.filter-pill').forEach(p => p.classList.remove('active'));
      document.querySelector('.filter-pill[data-filter="all"]')?.classList.add('active');
      await loadDatasetImages(state.activeDatasetId);
      const newIdx = state.images.findIndex(img => img.filename === nextItem.filename);
      if (newIdx >= 0) selectImage(newIdx);
    }
  } catch (err) {
    alert('Failed to load next difficult image: ' + err.message);
  } finally {
    hideSpinner();
  }
}

/**
 * Load the active learning priority queue into the sidebar.
 * Calls GET /api/datasets/{dataset_id}/active_learning
 */
async function loadActiveLearningQueue() {
  if (!state.activeDatasetId) return;
  try {
    const filterVal = elements.alFilterSelect.value;
    const filterParam = filterVal && filterVal !== 'all' ? `?filter_reason=${filterVal}` : '';
    const res = await fetch(`/api/datasets/${state.activeDatasetId}/active_learning${filterParam}`);
    if (!res.ok) return;
    const data = await res.json();
    const queue = data.queue || [];
    
    // Update count badge
    elements.alCountBadge.textContent = queue.length;
    
    // Render queue list
    if (queue.length === 0) {
      elements.alQueueList.innerHTML = '<div class="al-empty">No difficult examples pending review.</div>';
      return;
    }
    
    elements.alQueueList.innerHTML = '';
    // Show top 20 items max in sidebar
    const displayItems = queue.slice(0, 20);
    displayItems.forEach(item => {
      const row = document.createElement('div');
      row.className = 'al-queue-item';
      
      const priorityClass = item.priority >= 70 ? 'high' : item.priority >= 40 ? 'medium' : 'low';
      const reasons = (item.difficulty_reasons || []).slice(0, 2).join(', ');
      
      row.innerHTML = `
        <div class="al-item-info">
          <span class="al-item-filename" title="${item.filename}">${item.filename}</span>
          <span class="al-item-reasons">${reasons || 'needs review'}</span>
        </div>
        <span class="al-priority-badge ${priorityClass}">${item.priority}</span>
      `;
      
      // Click to navigate to that image
      row.addEventListener('click', () => {
        const idx = state.images.findIndex(img => img.filename === item.filename);
        if (idx >= 0) selectImage(idx);
      });
      
      elements.alQueueList.appendChild(row);
    });
    
    if (queue.length > 20) {
      const moreRow = document.createElement('div');
      moreRow.className = 'al-queue-item al-more';
      moreRow.textContent = `+ ${queue.length - 20} more items`;
      elements.alQueueList.appendChild(moreRow);
    }
  } catch (err) {
    console.error('Failed to load active learning queue:', err);
  }
}

/**
 * Export hard cases subset.
 * Calls POST /api/datasets/{dataset_id}/export_hard_cases
 */
async function exportHardCases() {
  if (!state.activeDatasetId) return;
  showSpinner('Exporting hard cases...');
  try {
    const res = await fetch(`/api/datasets/${state.activeDatasetId}/export_hard_cases?min_priority=60`, {
      method: 'POST'
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const result = data.result;
    alert(`Hard Cases Export Complete!\n\nExport Path: ${result.export_path || 'N/A'}\nImages: ${result.exported_count || 0}`);
  } catch (err) {
    alert('Export hard cases failed: ' + err.message);
  } finally {
    hideSpinner();
  }
}

// --- DATASET EXPORT ---

async function exportDataset() {
  showSpinner('Exporting YOLO-OBB dataset...');
  try {
    const res = await fetch(`/api/datasets/${state.activeDatasetId}/export`, { method: 'POST' });
    const data = await res.json();
    alert(`YOLO-OBB Export Successful!\n\nExport Path: ${data.result.export_path}\nImages Exported: ${data.result.exported_images}\nTrain: ${data.result.train_count}, Val: ${data.result.val_count}`);
  } catch (err) {
    alert('Export failed: ' + err.message);
  } finally {
    hideSpinner();
  }
}

// --- BATCH AUTO-LABELING ---

async function startBatchJob() {
  elements.btnStartBatch.disabled = true;
  try {
    const res = await fetch('/api/batch/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset_id: state.activeDatasetId,
        mode: elements.batchModeSelect.value,
        conf_threshold: 0.25,
        only_unlabeled: elements.batchUnlabeledOnly.checked,
        use_sam_refinement: elements.batchSamRefine.checked
      })
    });
    const data = await res.json();
    
    elements.btnPauseBatch.disabled = false;
    elements.btnCancelBatch.disabled = false;
    
    if (state.batchPollInterval) clearInterval(state.batchPollInterval);
    state.batchPollInterval = setInterval(pollBatchStatus, 1000);
  } catch (err) {
    alert('Failed to start batch job: ' + err.message);
    elements.btnStartBatch.disabled = false;
  }
}

async function pollBatchStatus() {
  try {
    const res = await fetch('/api/batch/status');
    const data = await res.json();
    const job = data.job;
    
    elements.batchStatusVal.textContent = job.status.toUpperCase();
    elements.batchProgressVal.textContent = `${job.processed_count} / ${job.total_images}`;
    elements.batchAcceptedVal.textContent = job.accepted_count;
    elements.batchReviewVal.textContent = job.needs_review_count;
    elements.batchFailedVal.textContent = job.failed_count;
    
    const pct = job.total_images > 0 ? (job.processed_count / job.total_images) * 100 : 0;
    elements.batchProgressFill.style.width = `${pct}%`;
    
    if (job.status === 'completed' || job.status === 'cancelled' || job.status === 'idle') {
      clearInterval(state.batchPollInterval);
      state.batchPollInterval = null;
      elements.btnStartBatch.disabled = false;
      elements.btnPauseBatch.disabled = true;
      elements.btnResumeBatch.disabled = true;
      elements.btnCancelBatch.disabled = true;
      await loadDatasetImages(state.activeDatasetId);
    }
  } catch (err) {
    console.error('Error polling batch status:', err);
  }
}

// --- IMPORT IMAGES ---

async function runImageImport() {
  const dirPath = elements.importSourcePath.value.trim();
  if (!dirPath) {
    alert('Please enter a valid directory path.');
    return;
  }
  showSpinner('Importing images...');
  try {
    const res = await fetch(`/api/datasets/${state.activeDatasetId}/import`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source_dir: dirPath })
    });
    const data = await res.json();
    alert(`Imported ${data.imported_count} images.`);
    elements.importModal.classList.add('hidden');
    await loadDatasetImages(state.activeDatasetId);
  } catch (err) {
    alert('Import failed: ' + err.message);
  } finally {
    hideSpinner();
  }
}

// --- KEYBOARD SHORTCUTS ---

function handleKeyDown(e) {
  // Ignore shortcuts when typing in an input
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
  
  if (e.key === 'a' || e.key === 'A') {
    e.preventDefault();
    acceptCurrentAnnotation();
  } else if (e.key === 'r' || e.key === 'R') {
    e.preventDefault();
    rejectCurrentAnnotation();
  } else if (e.key === ' ' || e.key === 'Enter') {
    e.preventDefault();
    saveEditsCurrentAnnotation();
  } else if (e.key === 's' || e.key === 'S') {
    refineSelectedWithSAM();
  } else if (e.key === 'd' || e.key === 'D' || e.key === 'Delete') {
    deleteSelectedBox();
  } else if (e.key === 'w' || e.key === 'W') {
    setToolMode('draw');
  } else if (e.key === 'v' || e.key === 'V' || e.key === 'Escape') {
    setToolMode('select');
  } else if (e.key === 'n' || e.key === 'N' || e.key === 'ArrowRight') {
    selectImage(state.activeImageIndex + 1);
  } else if (e.key === 'p' || e.key === 'P' || e.key === 'ArrowLeft') {
    selectImage(state.activeImageIndex - 1);
  } else if (e.key === 'f' || e.key === 'F') {
    zoomFit();
  } else if (e.key === '+') {
    zoomBy(1.2);
  } else if (e.key === '-') {
    zoomBy(0.8);
  } else if (e.key === '?') {
    elements.shortcutsModal.classList.toggle('hidden');
  } else if (e.key === 'q' || e.key === 'Q') {
    loadNextDifficultImage();
  }
}

// --- SPINNER HELPERS ---

function showSpinner(text) {
  elements.spinnerText.textContent = text || 'Processing...';
  elements.canvasSpinner.classList.remove('hidden');
}

function hideSpinner() {
  elements.canvasSpinner.classList.add('hidden');
}

// Initialize on page load
window.addEventListener('DOMContentLoaded', init);
