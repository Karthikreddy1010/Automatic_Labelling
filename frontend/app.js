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

// Base URL for all backend calls, derived from this script's own resolved
// URL rather than hardcoded as "/". Works identically whether the app is
// served at the domain root or behind a reverse-proxy sub-path (JupyterHub's
// /proxy/, VS Code Server's /vscode/proxy/, etc.) with no environment-specific
// configuration -- a hardcoded "/api/..." path breaks under any such prefix
// because the browser resolves it against the domain root, not the proxy path.
const API_BASE = document.currentScript
  ? new URL('.', document.currentScript.src).href
  : window.location.origin + '/';

// Bumped every time selectImage() targets a new image. Async handlers
// (image load, annotation/prediction fetches, AI inference, SAM refine)
// capture it before their await and re-check it after, so a response that
// arrives after the user has navigated away is discarded instead of being
// applied to whatever image is now active.
let activeLoadToken = 0;

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
  labelFilter: 'all',   // 'all' | 'accept' | 'review' | 'reject' | 'disagreement' | 'geometry_warning' -- filters the per-image Labels list only

  batchPollInterval: null,
  activeDevice: 'AUTO',

  // Client-side undo/redo: snapshots of state.boxes, reset per-image.
  history: [],
  historyIndex: -1,
  isDirty: false,

  // Import modal staging queue: [{ id, file, url, status: 'pending'|'uploading'|'done'|'error', el }]
  importQueue: [],
  importUploading: false,
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
  btnUndo: document.getElementById('btn-undo'),
  btnRedo: document.getElementById('btn-redo'),
  
  toggleYolo: document.getElementById('toggle-yolo'),
  toggleDino: document.getElementById('toggle-dino'),
  toggleSam: document.getElementById('toggle-sam'),
  
  btnPrevImg: document.getElementById('btn-prev-img'),
  btnNextImg: document.getElementById('btn-next-img'),
  paginationText: document.getElementById('image-pagination-text'),
  imageStatusBadge: document.getElementById('image-status-badge'),
  unsavedIndicator: document.getElementById('unsaved-indicator'),
  btnAcceptVerify: document.getElementById('btn-accept-verify'),
  btnAccept: document.getElementById('btn-accept'),
  btnReject: document.getElementById('btn-reject'),
  btnSkip: document.getElementById('btn-skip'),
  btnRemoveAllLabels: document.getElementById('btn-remove-all-labels'),
  btnNextDifficult: document.getElementById('btn-next-difficult'),

  // Labels List
  labelsCountBadge: document.getElementById('labels-count-badge'),
  labelsList: document.getElementById('labels-list'),
  labelsFilterChips: document.getElementById('labels-filter-chips'),

  // Provenance Elements
  provPipeline: document.getElementById('prov-pipeline'),
  provQwen: document.getElementById('prov-qwen'),
  provQwenMaterialRow: document.getElementById('prov-qwen-material-row'),
  provQwenMaterialVal: document.getElementById('prov-qwen-material-val'),
  provQwenVisibilityRow: document.getElementById('prov-qwen-visibility-row'),
  provQwenVisibilityVal: document.getElementById('prov-qwen-visibility-val'),
  provDecisionRow: document.getElementById('prov-decision-row'),
  provDecisionBadge: document.getElementById('prov-decision-badge'),
  provFinalScoreRow: document.getElementById('prov-final-score-row'),
  provFinalScoreVal: document.getElementById('prov-final-score-val'),
  provGeometryRow: document.getElementById('prov-geometry-row'),
  provGeometryVal: document.getElementById('prov-geometry-val'),
  provSegmentationRow: document.getElementById('prov-segmentation-row'),
  provSegmentationVal: document.getElementById('prov-segmentation-val'),
  provDisagreementRow: document.getElementById('prov-disagreement-row'),
  provDisagreementVal: document.getElementById('prov-disagreement-val'),
  provHumanStatus: document.getElementById('prov-human-status'),
  provQualityRow: document.getElementById('prov-quality-row'),
  provQualityVal: document.getElementById('prov-quality-val'),
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
  batchGeometryQa: document.getElementById('batch-geometry-qa'),
  batchQwenGating: document.getElementById('batch-qwen-gating'),
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
  importDropzone: document.getElementById('import-dropzone'),
  importFolderInput: document.getElementById('import-folder-input'),
  importFilesInput: document.getElementById('import-files-input'),
  btnBrowseFolder: document.getElementById('btn-browse-folder'),
  btnBrowseFiles: document.getElementById('btn-browse-files'),
  importStaging: document.getElementById('import-staging'),
  importStagingCount: document.getElementById('import-staging-count'),
  btnClearStaging: document.getElementById('btn-clear-staging'),
  importThumbGrid: document.getElementById('import-thumb-grid'),
  btnStartUpload: document.getElementById('btn-start-upload'),

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

// Enclosing axis-aligned box for a set of corners. box.xyxy must be kept in
// sync with box.corners any time corners are mutated after creation (moved,
// resized, rotated, corner-dragged, or replaced by a SAM refine) -- a few
// features (e.g. "Refine with SAM") read box.xyxy directly, and a stale
// value would re-segment the box's original position instead of its current one.
function cornersToXyxy(corners) {
  const xs = corners.map(p => p[0]);
  const ys = corners.map(p => p[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
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
    const res = await fetch(API_BASE + 'api/system/hardware');
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
    const res = await fetch(API_BASE + 'api/models/status');
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
        <span class="m-name">${m.name}${m.backend ? `<span class="model-backend">${m.backend}${m.device && m.device !== 'cpu' ? ' · ' + m.device : ''}</span>` : ''}</span>
        <span class="m-badge ${badgeClass}" style="cursor: pointer;" title="Status: ${m.status}. Click to pre-load into memory.">${labelText}</span>
      `;
      
      // Allow user to click badge to pre-load model
      const badge = row.querySelector('.m-badge');
      if (m.status === 'not_loaded') {
        badge.addEventListener('click', async () => {
          badge.textContent = 'loading...';
          try {
            await fetch(API_BASE + `api/models/load?model_id=${encodeURIComponent(m.id)}`, { method: 'POST' });
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
    const data = await fetchJson(API_BASE + 'api/datasets');
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
    elements.paginationText.textContent = `Failed to load datasets: ${err.message}`;
  }
}

/**
 * Prompt for a new dataset name (bound to the "+" button next to the
 * dataset dropdown) and create it. Guards against overwriting an existing
 * dataset's metadata: create_dataset() on the backend silently resets
 * metadata.json (image_count/status tracking) if the sanitized ID already
 * exists, so a same-named collision is caught here first rather than
 * silently wiping a dataset the user didn't mean to touch.
 */
async function promptCreateDataset() {
  const name = prompt('New dataset name:');
  if (!name || !name.trim()) return;

  const id = name.trim().toLowerCase().replace(/[^a-z0-9_-]+/g, '_');
  if (state.datasets.some(ds => ds.dataset_id === id)) {
    alert(`A dataset with id "${id}" already exists. Choose a different name.`);
    return;
  }

  await createNewDataset(id, name.trim());
}

async function createNewDataset(id, name) {
  try {
    const res = await fetch(API_BASE + 'api/datasets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset_id: id, name: name, classes: ['utility_pole'] })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    await loadDatasets();
  } catch (err) {
    alert('Failed to create dataset: ' + err.message);
  }
}

async function loadDatasetImages(datasetId) {
  try {
    const filterParam = state.filterStatus === 'all' ? '' : `?status=${state.filterStatus}`;
    const data = await fetchJson(API_BASE + `api/datasets/${datasetId}/images${filterParam}`);
    state.images = data.images || [];

    // Update count badge & stats
    const dsData = await fetchJson(API_BASE + `api/datasets/${datasetId}`);
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
    elements.paginationText.textContent = `Failed to load images: ${err.message}`;
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
  if (state.isDirty) {
    const proceed = confirm('You have unsaved changes on this image. Leave without saving?');
    if (!proceed) return;
  }
  const token = ++activeLoadToken;
  state.activeImageIndex = index;
  state.activeImageMeta = state.images[index];
  state.selectedBoxIndex = -1;
  updateSelectionInspector();
  updateTestSetUI(!!state.activeImageMeta.is_test_set);
  updateImageStatusBadge();
  setPositiveTags(state.activeImageMeta.al_categories || []);
  renderGallery();

  elements.paginationText.textContent = `${index + 1} / ${state.images.length} — ${state.activeImageMeta.filename}`;

  showSpinner('Loading image & labels...');

  // 1. Load image onto HTML Image element
  const imgUrl = `${API_BASE}api/datasets/${state.activeDatasetId}/images/${encodeURIComponent(state.activeImageMeta.filename)}`;
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.src = imgUrl;
  const loadOk = await new Promise((resolve) => {
    img.onload = () => resolve(true);
    img.onerror = () => { console.error('Error loading image:', imgUrl); resolve(false); };
  });
  if (token !== activeLoadToken) return; // superseded by a newer selectImage() call
  state.imageObj = loadOk ? img : null;
  state.imageLoadFailed = !loadOk;

  // 2. Fetch existing annotations or predictions
  let loadedBoxes = [];
  try {
    const annData = await fetchJson(API_BASE + `api/datasets/${state.activeDatasetId}/images/${encodeURIComponent(state.activeImageMeta.filename)}/annotations`);

    if (annData.data && annData.data.boxes) {
      loadedBoxes = annData.data.boxes.map(b => {
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
  if (token !== activeLoadToken) return; // superseded by a newer selectImage() call
  state.boxes = loadedBoxes;
  initHistory();
  setDirty(false);
  renderLabelList();

  // 3. Fetch differential history for this image
  await fetchDifferentialHistory(state.activeImageMeta.filename);
  if (token !== activeLoadToken) return; // superseded by a newer selectImage() call

  hideSpinner();
  zoomFit();
}

async function fetchDifferentialHistory(filename) {
  try {
    const res = await fetch(API_BASE + `api/datasets/${state.activeDatasetId}/history?filename=${encodeURIComponent(filename)}`);
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

  if (!state.imageObj) {
    // Never leave the canvas silently black/blank -- tell the user why.
    ctx.save();
    ctx.fillStyle = '#8b949e';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    const msg = state.imageLoadFailed
      ? 'Failed to load this image (check the file exists and the API/proxy connection).'
      : 'No image loaded.';
    ctx.fillText(msg, rect.width / 2, rect.height / 2);
    ctx.restore();
    return;
  }

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
  const decision = box.attributes && box.attributes.decision; // 'ACCEPT'|'REVIEW'|'REJECT' from decision_engine.py, when this box went through it

  if (box.model_source !== 'HUMAN') {
    if (decision === 'ACCEPT') {
      strokeColor = '#2ea043'; fillColor = 'rgba(46, 160, 67, 0.2)';
    } else if (decision === 'REJECT') {
      strokeColor = '#f85149'; fillColor = 'rgba(248, 81, 73, 0.2)';
    } else if (decision === 'REVIEW') {
      strokeColor = '#d29922'; fillColor = 'rgba(210, 153, 34, 0.2)';
    } else if (box.needs_review || box.confidence < 0.45) {
      strokeColor = '#f85149'; // Red (review / low conf) -- no decision engine output (e.g. YOLO_FAST/RUN_ALL)
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
    strokeColor = '#8b5cf6'; // Accent purple highlight for selected (matches CSS --color-accent)
    fillColor = 'rgba(139, 92, 246, 0.25)';
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
    ctx.strokeStyle = '#8b5cf6';
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(rotX, rotY, handleRadius * 1.1, 0, Math.PI * 2);
    ctx.fillStyle = '#8b5cf6';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.stroke();
  }
  
  // Draw Label Badge
  const badgeX = corners[0][0];
  const badgeY = corners[0][1] - 8 / state.transform.scale;
  ctx.font = `${Math.max(10, 11 / state.transform.scale)}px ${getComputedStyle(document.body).fontFamily}`;
  const confText = box.confidence ? `${Math.round(box.confidence * 100)}%` : '';
  const className = box.class_name || 'utility_pole';
  let labelStr;
  if (box.model_source === 'HUMAN') {
    labelStr = `${className} · Human`;
  } else if (decision) {
    labelStr = `${className} · ${decision} ${confText}`;
  } else {
    labelStr = `${className} · ${confText}`;
  }
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
  ctx.strokeStyle = '#8b5cf6';
  ctx.lineWidth = 1.5 / state.transform.scale;
  ctx.fillStyle = 'rgba(139, 92, 246, 0.15)';
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
  elements.btnSkip.addEventListener('click', skipCurrentAnnotation);
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

  // Undo/Redo & Remove All Labels
  if (elements.btnUndo) elements.btnUndo.addEventListener('click', undo);
  if (elements.btnRedo) elements.btnRedo.addEventListener('click', redo);
  if (elements.btnRemoveAllLabels) elements.btnRemoveAllLabels.addEventListener('click', removeAllLabelsCurrentImage);

  // Labels list filter chips (accept/review/reject/disagreement/geometry warning)
  if (elements.labelsFilterChips) {
    elements.labelsFilterChips.querySelectorAll('.filter-chip').forEach(chip => {
      chip.addEventListener('click', () => setLabelFilter(chip.dataset.filter));
    });
  }

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
    await fetch(API_BASE + 'api/system/device', {
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
  elements.btnPauseBatch.addEventListener('click', () => fetch(API_BASE + 'api/batch/pause', { method: 'POST' }));
  elements.btnResumeBatch.addEventListener('click', () => fetch(API_BASE + 'api/batch/resume', { method: 'POST' }));
  elements.btnCancelBatch.addEventListener('click', () => fetch(API_BASE + 'api/batch/cancel', { method: 'POST' }));
  
  elements.btnNewDataset.addEventListener('click', promptCreateDataset);
  elements.btnImportImages.addEventListener('click', () => elements.importModal.classList.remove('hidden'));
  elements.btnCloseImportModal.addEventListener('click', closeImportModal);
  elements.btnCancelImport.addEventListener('click', closeImportModal);
  elements.btnRunImport.addEventListener('click', runImageImport);

  elements.btnBrowseFolder.addEventListener('click', () => elements.importFolderInput.click());
  elements.btnBrowseFiles.addEventListener('click', () => elements.importFilesInput.click());
  elements.importFolderInput.addEventListener('change', (e) => {
    addFilesToStagingQueue(e.target.files);
    e.target.value = '';
  });
  elements.importFilesInput.addEventListener('change', (e) => {
    addFilesToStagingQueue(e.target.files);
    e.target.value = '';
  });
  ['dragenter', 'dragover'].forEach(evt =>
    elements.importDropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      elements.importDropzone.classList.add('dragging');
    })
  );
  ['dragleave', 'drop'].forEach(evt =>
    elements.importDropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      elements.importDropzone.classList.remove('dragging');
    })
  );
  elements.importDropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files.length) addFilesToStagingQueue(e.dataTransfer.files);
  });
  elements.btnClearStaging.addEventListener('click', clearStagingQueue);
  elements.btnStartUpload.addEventListener('click', startStagedUpload);

  elements.btnShortcuts.addEventListener('click', () => elements.shortcutsModal.classList.toggle('hidden'));
  elements.btnCloseShortcutsModal.addEventListener('click', () => elements.shortcutsModal.classList.add('hidden'));
  
  // Keyboard Shortcuts
  window.addEventListener('keydown', handleKeyDown);

  // Warn before closing/reloading the tab with unsaved edits.
  window.addEventListener('beforeunload', (e) => {
    if (state.isDirty) {
      e.preventDefault();
      e.returnValue = '';
    }
  });
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
      pushHistory();
      updateSelectionInspector();
    }
    state.isDrawing = false;
    state.drawStart = null;
    render();
  }

  if (state.isDraggingHandle && state.dragTarget) {
    // Canonicalize corners on drag release
    const box = state.dragTarget.box;
    const moved = JSON.stringify(box.corners) !== JSON.stringify(state.dragTarget.origCorners);
    box.corners = orderCornersCanonical(box.corners);
    box.xyxy = cornersToXyxy(box.corners);
    state.isDraggingHandle = false;
    state.dragTarget = null;
    if (moved) pushHistory(); // one undo step per drag, not per mousemove; skip no-op clicks
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
    renderLabelList();
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
  // Pipeline source chain -- built from what actually ran for THIS candidate
  // (box.attributes), not just guessed from the model_source string, so it
  // stays accurate as the pipeline gains/loses stages.
  const srcLower = src.toLowerCase();
  let pipelineText = src;
  if (srcLower.includes('dino') && srcLower.includes('sam')) {
    const stages = ['Grounding DINO', 'SAM3', 'SAM refine'];
    if (box.attributes && box.attributes.geometry_status) stages.push('Geometry QA');
    stages.push(box.attributes && box.attributes.qwen_ran ? 'Qwen' : 'Qwen (skipped)');
    stages.push('Decision');
    pipelineText = stages.join(' → ');
  } else if (srcLower.includes('dino')) {
    pipelineText = 'Grounding DINO → OBB';
  } else if (srcLower.includes('sam')) {
    pipelineText = 'SAM → OBB';
  } else if (srcLower.includes('yolo')) {
    pipelineText = 'YOLO-OBB';
  } else if (srcLower === 'human') {
    pipelineText = 'Human Annotation';
  }
  elements.provPipeline.textContent = pipelineText;

  const attrs = box.attributes || {};

  // Decision badge (ACCEPT/REVIEW/REJECT from the multi-signal decision engine).
  // A human can always override this via edit/Accept/Reject -- it's advisory,
  // not a lock.
  if (attrs.decision) {
    elements.provDecisionRow.classList.remove('hidden');
    elements.provDecisionBadge.textContent = attrs.decision;
    elements.provDecisionBadge.className = 'badge ' + (
      attrs.decision === 'ACCEPT' ? 'badge-success' : attrs.decision === 'REJECT' ? 'badge-danger' : 'badge-warning'
    );
  } else {
    elements.provDecisionRow.classList.add('hidden');
  }

  if (typeof attrs.final_score === 'number') {
    elements.provFinalScoreRow.classList.remove('hidden');
    elements.provFinalScoreVal.textContent = attrs.final_score.toFixed(2);
  } else {
    elements.provFinalScoreRow.classList.add('hidden');
  }

  // Qwen semantic verification (from box attributes if it ran for this candidate)
  if (attrs.qwen_ran && attrs.qwen_class) {
    const conf = typeof attrs.qwen_semantic_confidence === 'number' ? ` (${attrs.qwen_semantic_confidence.toFixed(2)})` : '';
    elements.provQwen.textContent = `${attrs.qwen_class.replace(/_/g, ' ')}${conf}`;
    elements.provQwen.className = 'prov-v ' + (attrs.qwen_class === 'electric_utility_pole' ? 'green' : 'red');
  } else {
    elements.provQwen.textContent = 'Not run';
    elements.provQwen.className = 'prov-v yellow';
  }
  if (attrs.qwen_ran && attrs.qwen_material) {
    elements.provQwenMaterialRow.classList.remove('hidden');
    elements.provQwenMaterialVal.textContent = attrs.qwen_material;
  } else {
    elements.provQwenMaterialRow.classList.add('hidden');
  }
  if (attrs.qwen_ran && attrs.qwen_visibility) {
    elements.provQwenVisibilityRow.classList.remove('hidden');
    elements.provQwenVisibilityVal.textContent = attrs.qwen_visibility.replace(/_/g, ' ');
  } else {
    elements.provQwenVisibilityRow.classList.add('hidden');
  }

  // Geometry QA (geometry_qa.py) -- score + status, not just a pass/fail flag.
  if (attrs.geometry_status) {
    elements.provGeometryRow.classList.remove('hidden');
    const gScore = typeof attrs.geometry_score === 'number' ? attrs.geometry_score.toFixed(2) : '--';
    elements.provGeometryVal.textContent = `${attrs.geometry_status} (${gScore})`;
    elements.provGeometryVal.className = 'prov-v ' + (
      attrs.geometry_status === 'pass' ? 'green' : attrs.geometry_status === 'fail' ? 'red' : 'yellow'
    );
    elements.provGeometryVal.title = (attrs.geometry_flags || []).join('; ');
  } else {
    elements.provGeometryRow.classList.add('hidden');
  }

  if (typeof attrs.segmentation_score === 'number') {
    elements.provSegmentationRow.classList.remove('hidden');
    elements.provSegmentationVal.textContent = attrs.segmentation_score.toFixed(2);
  } else {
    elements.provSegmentationRow.classList.add('hidden');
  }

  // Disagreement between independent signals (DINO vs Qwen, Qwen vs SAM, etc.)
  if (attrs.disagreements && attrs.disagreements.length > 0) {
    elements.provDisagreementRow.classList.remove('hidden');
    elements.provDisagreementVal.textContent = attrs.disagreements.join('; ');
  } else {
    elements.provDisagreementRow.classList.add('hidden');
  }

  // Human review status
  const imgStatus = state.activeImageMeta ? (state.activeImageMeta.status || 'unlabeled') : 'unlabeled';
  elements.provHumanStatus.textContent = imgStatus.replace('_', ' ');
  elements.provHumanStatus.className = `prov-v ${imgStatus === 'accepted' || imgStatus === 'verified' ? 'green' : imgStatus === 'rejected' ? 'red' : 'yellow'}`;
  
  // Quality score (heuristic signal from the DINO+SAM pipeline, NOT a
  // calibrated probability -- labeled and treated as such).
  if (box.attributes && typeof box.attributes.quality_score === 'number') {
    elements.provQualityRow.classList.remove('hidden');
    const q = box.attributes.quality_score;
    const cat = box.attributes.category || '';
    elements.provQualityVal.textContent = `${q.toFixed(2)}${cat ? ' (' + cat.replace('_', ' ') + ')' : ''}`;
    elements.provQualityVal.className = `prov-v ${cat === 'HIGH_QUALITY' ? 'green' : cat === 'REJECT' ? 'red' : 'yellow'}`;
  } else {
    elements.provQualityRow.classList.add('hidden');
  }

  // Correction IoU (from box attributes if available)
  if (box.attributes && box.attributes.correction_iou !== undefined) {
    elements.provIouRow.classList.remove('hidden');
    const iou = (box.attributes.correction_iou * 100).toFixed(1);
    elements.provIouVal.textContent = `${iou}%`;
  } else {
    elements.provIouRow.classList.add('hidden');
    elements.provIouVal.textContent = '--';
  }

  // Active Learning priority (from image meta)
  if (state.activeImageMeta && state.activeImageMeta.priority !== undefined) {
    const priority = state.activeImageMeta.priority;
    elements.provPriorityRow.classList.remove('hidden');
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

  renderLabelList();
}

function deleteSelectedBox() {
  deleteBoxAtIndex(state.selectedBoxIndex);
}

/**
 * Remove a single label by index -- the underlying operation for both the
 * Inspector panel's Delete button (deleteSelectedBox, uses the currently
 * selected box) and each Labels-list row's own inline delete button (lets
 * you remove one of several poles directly from the list, without first
 * having to find and click its OBB on the canvas).
 */
function deleteBoxAtIndex(idx) {
  if (idx < 0 || idx >= state.boxes.length) return;
  state.boxes.splice(idx, 1);
  if (state.selectedBoxIndex === idx) {
    state.selectedBoxIndex = -1;
  } else if (state.selectedBoxIndex > idx) {
    state.selectedBoxIndex -= 1;
  }
  pushHistory();
  updateSelectionInspector();
  renderLabelList();
  render();
}

function removeAllLabelsCurrentImage() {
  if (!state.activeImageMeta) return;
  if (state.boxes.length === 0) return;
  const confirmed = confirm(
    'Remove all labels from this image?\n\n' +
    'This will remove all labels from the current image only. ' +
    'Click "Save Edits" afterward to persist this change.'
  );
  if (!confirmed) return;
  state.boxes = [];
  state.selectedBoxIndex = -1;
  pushHistory();
  updateSelectionInspector();
  render();
}

// --- UNDO / REDO & UNSAVED-CHANGES TRACKING ---
// Client-side snapshot history of state.boxes, reset per image. Every
// committed edit (add/delete/move/resize/rotate/corner-drag/clear-all) pushes
// one snapshot; continuous drag operations only push once, on mouseup, so a
// single drag is a single undo step.

function cloneBoxes(boxes) {
  return JSON.parse(JSON.stringify(boxes || []));
}

function initHistory() {
  state.history = [cloneBoxes(state.boxes)];
  state.historyIndex = 0;
  updateUndoRedoButtons();
}

function pushHistory() {
  state.history = state.history.slice(0, state.historyIndex + 1);
  state.history.push(cloneBoxes(state.boxes));
  state.historyIndex = state.history.length - 1;
  const MAX_HISTORY = 100;
  if (state.history.length > MAX_HISTORY) {
    state.history.shift();
    state.historyIndex--;
  }
  updateUndoRedoButtons();
  setDirty(true);
}

function undo() {
  if (state.historyIndex <= 0) return;
  state.historyIndex--;
  state.boxes = cloneBoxes(state.history[state.historyIndex]);
  state.selectedBoxIndex = -1;
  updateUndoRedoButtons();
  updateSelectionInspector();
  render();
  setDirty(state.historyIndex !== 0);
}

function redo() {
  if (state.historyIndex >= state.history.length - 1) return;
  state.historyIndex++;
  state.boxes = cloneBoxes(state.history[state.historyIndex]);
  state.selectedBoxIndex = -1;
  updateUndoRedoButtons();
  updateSelectionInspector();
  render();
  setDirty(state.historyIndex !== 0);
}

function updateUndoRedoButtons() {
  if (elements.btnUndo) elements.btnUndo.disabled = state.historyIndex <= 0;
  if (elements.btnRedo) elements.btnRedo.disabled = state.historyIndex >= state.history.length - 1;
}

function setDirty(value) {
  state.isDirty = value;
  if (elements.unsavedIndicator) elements.unsavedIndicator.classList.toggle('hidden', !value);
}

// --- LABEL LIST (Pole 1 / Pole 2 / ... sidebar, synced with canvas selection) ---

// Whether a box matches the active labels-list filter chip. Filters only
// change what's shown in this list -- they never delete/hide boxes on the
// canvas or affect what gets saved.
function boxMatchesLabelFilter(box, filter) {
  if (filter === 'all') return true;
  const attrs = box.attributes || {};
  const decision = attrs.decision; // 'ACCEPT' | 'REVIEW' | 'REJECT' | undefined
  if (filter === 'accept') return decision === 'ACCEPT' || (!decision && !box.needs_review);
  if (filter === 'review') return decision === 'REVIEW' || (!decision && box.needs_review);
  if (filter === 'reject') return decision === 'REJECT';
  if (filter === 'disagreement') return !!(attrs.disagreements && attrs.disagreements.length > 0);
  if (filter === 'geometry_warning') return attrs.geometry_status === 'warning' || attrs.geometry_status === 'fail';
  return true;
}

function setLabelFilter(filter) {
  state.labelFilter = filter;
  if (elements.labelsFilterChips) {
    elements.labelsFilterChips.querySelectorAll('.filter-chip').forEach(chip => {
      chip.classList.toggle('active', chip.dataset.filter === filter);
    });
  }
  renderLabelList();
}

function renderLabelList() {
  if (!elements.labelsList) return;
  const boxes = state.boxes || [];
  if (elements.labelsCountBadge) elements.labelsCountBadge.textContent = boxes.length;

  if (boxes.length === 0) {
    elements.labelsList.innerHTML = '<div class="labels-empty">No labels on this image.</div>';
    return;
  }

  const filter = state.labelFilter || 'all';
  const visibleIndices = boxes
    .map((box, idx) => ({ box, idx }))
    .filter(({ box }) => boxMatchesLabelFilter(box, filter));

  if (visibleIndices.length === 0) {
    elements.labelsList.innerHTML = `<div class="labels-empty">No labels match filter "${filter.replace('_', ' ')}".</div>`;
    return;
  }

  elements.labelsList.innerHTML = '';
  visibleIndices.forEach(({ box, idx }) => {
    const row = document.createElement('div');
    row.className = 'label-row' + (idx === state.selectedBoxIndex ? ' selected' : '');

    const attrs = box.attributes || {};
    const isHuman = (box.model_source || '').toUpperCase() === 'HUMAN';
    const sourceLabel = isHuman ? 'Human' : (box.model_source || 'AI');
    const decision = attrs.decision;
    const statusLabel = decision || (box.needs_review ? 'Review' : 'Verified');
    const statusClass = decision === 'ACCEPT' ? 'verified' : decision === 'REJECT' ? 'rejected'
      : decision === 'REVIEW' ? 'review' : (box.needs_review ? 'review' : 'verified');
    const hasQuality = typeof attrs.quality_score === 'number';
    const qualityText = hasQuality ? `Q: ${attrs.quality_score.toFixed(2)}` : '';
    const hasDisagreement = attrs.disagreements && attrs.disagreements.length > 0;

    const nameEl = document.createElement('span');
    nameEl.className = 'label-row-name';
    nameEl.textContent = `Pole ${idx + 1}`;

    const sourceEl = document.createElement('span');
    sourceEl.className = 'label-row-source' + (isHuman ? ' human' : '');
    sourceEl.textContent = sourceLabel;

    row.appendChild(nameEl);
    row.appendChild(sourceEl);
    if (qualityText) {
      const qEl = document.createElement('span');
      qEl.className = 'label-row-quality';
      qEl.textContent = qualityText;
      row.appendChild(qEl);
    }
    if (hasDisagreement) {
      const dEl = document.createElement('span');
      dEl.className = 'label-row-disagreement';
      dEl.textContent = '⚠ disagreement';
      dEl.title = attrs.disagreements.join('; ');
      row.appendChild(dEl);
    }
    const statusEl = document.createElement('span');
    statusEl.className = `label-row-status ${statusClass}`;
    statusEl.textContent = statusLabel;
    row.appendChild(statusEl);

    const removeBtn = document.createElement('button');
    removeBtn.className = 'label-row-remove';
    removeBtn.type = 'button';
    removeBtn.title = `Remove Pole ${idx + 1}`;
    removeBtn.textContent = '×';
    removeBtn.addEventListener('click', (e) => {
      e.stopPropagation(); // don't also trigger the row's own select-on-click
      deleteBoxAtIndex(idx);
    });
    row.appendChild(removeBtn);

    row.addEventListener('click', () => {
      state.selectedBoxIndex = idx;
      updateSelectionInspector();
      render();
    });
    elements.labelsList.appendChild(row);
  });
}

// --- IMAGE STATUS BADGE ---

function updateImageStatusBadge() {
  if (!elements.imageStatusBadge) return;
  const meta = state.activeImageMeta;
  if (!meta) {
    elements.imageStatusBadge.classList.add('hidden');
    return;
  }
  const status = meta.status || 'unlabeled';
  const verifiedStatuses = ['verified', 'accepted', 'human_corrected'];
  let label = status.replace('_', ' ');
  let cssClass = status;
  if (meta.needs_review && !verifiedStatuses.includes(status)) {
    label = 'needs review';
    cssClass = 'needs_review';
  }
  elements.imageStatusBadge.textContent = label;
  elements.imageStatusBadge.className = `status-tag ${cssClass}`;
  elements.imageStatusBadge.classList.remove('hidden');
}

// --- SAFE FETCH: clear errors instead of "Unexpected token '<'" ---
// If the DeepThink proxy (or any reverse proxy) misroutes a request, the
// response is often an HTML error page, not JSON -- calling res.json() on
// that throws a cryptic SyntaxError. This checks content-type first and
// raises a message that actually explains what happened.

async function fetchJson(url, options) {
  let res;
  try {
    res = await fetch(url, options);
  } catch (netErr) {
    throw new Error(`Network error reaching the backend (${netErr.message}). Check the API/proxy connection.`);
  }
  const contentType = res.headers.get('content-type') || '';
  if (!contentType.includes('application/json')) {
    const text = await res.text().catch(() => '');
    throw new Error(
      `Server returned a non-JSON response (HTTP ${res.status}). This usually means the request ` +
      `didn't reach the backend (a proxy/routing issue), not an application error.` +
      (text ? ` Response started with: ${text.slice(0, 200).replace(/\s+/g, ' ')}` : '')
    );
  }
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || `Request failed (HTTP ${res.status})`);
  }
  return data;
}

// --- OBB COORDINATE VALIDATION (before save/export) ---

function validateBoxesForSave(boxes) {
  const errors = [];
  boxes.forEach((box, idx) => {
    const corners = box.corners;
    if (!Array.isArray(corners) || corners.length !== 4) {
      errors.push(`Pole ${idx + 1}: must have exactly 4 corners (has ${corners ? corners.length : 0}).`);
      return;
    }
    for (const pt of corners) {
      if (!Array.isArray(pt) || pt.length !== 2 || !Number.isFinite(pt[0]) || !Number.isFinite(pt[1])) {
        errors.push(`Pole ${idx + 1}: corner coordinates must be finite numbers (got ${JSON.stringify(pt)}).`);
        return;
      }
    }
    const area = Math.abs(signedShoelaceArea(corners));
    if (!(area > 1e-6)) {
      errors.push(`Pole ${idx + 1}: has zero/degenerate area -- drag its corners apart before saving.`);
    }
  });
  return errors;
}

/**
 * Clamp corners to the image bounds (poles may legitimately extend to/past
 * the frame edge -- clamp rather than reject) and return a deep copy safe to
 * send to the backend. Does not mutate the live editing state.
 */
function clampBoxesToImageBounds(boxes, imgWidth, imgHeight) {
  return boxes.map(box => ({
    ...box,
    corners: box.corners.map(([x, y]) => [
      Math.min(Math.max(x, 0), imgWidth),
      Math.min(Math.max(y, 0), imgHeight),
    ]),
  }));
}

// --- AI INFERENCE INTEGRATION ---

async function triggerInference(mode) {
  if (!state.activeImageMeta) return;
  const token = activeLoadToken; // snapshot: discard the result if the user navigates away before it lands
  const msg = mode === 'AI_LABEL'
    ? 'Running AI Label (DINO + SAM3 -> SAM refine -> Geometry QA -> Qwen -> Decision)...'
    : 'Running All Models (YOLO + Grounding DINO + SAM)... (~15s on CPU)';
  showSpinner(msg);
  try {
    const res = await fetch(API_BASE + 'api/inference/detect', {
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
    if (token !== activeLoadToken) return; // active image changed while this request was in flight
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
  const token = activeLoadToken; // snapshot: discard the result if the user navigates away before it lands
  const box = state.boxes[state.selectedBoxIndex];
  showSpinner('Refining with SAM segmentation...');
  try {
    let xyxy = box.xyxy;
    if (!xyxy && box.corners) {
      const xs = box.corners.map(p => p[0]);
      const ys = box.corners.map(p => p[1]);
      xyxy = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
    }

    const res = await fetch(API_BASE + 'api/inference/segment_box', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset_id: state.activeDatasetId,
        filename: state.activeImageMeta.filename,
        box_xyxy: xyxy
      })
    });
    const data = await res.json();
    // Bail if the active image changed, or this box is no longer the one
    // in state.boxes (e.g. a fresh AI inference run replaced the array
    // while this refine request was in flight) -- otherwise we'd either
    // corrupt the wrong image's boxes or silently mutate a detached object.
    if (token !== activeLoadToken || state.boxes[state.selectedBoxIndex] !== box) return;
    if (data.corners) {
      box.corners = orderCornersCanonical(data.corners);
      box.xyxy = cornersToXyxy(box.corners);
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
  const validationErrors = validateBoxesForSave(state.boxes);
  if (validationErrors.length > 0) {
    alert('Cannot save -- fix these OBBs first:\n\n' + validationErrors.join('\n'));
    return;
  }
  const tags = getSelectedPositiveTags();
  showSpinner('Accepting annotation as ground truth...');
  try {
    const clamped = clampBoxesToImageBounds(state.boxes, state.imageObj.width, state.imageObj.height);
    await fetchJson(API_BASE + `api/datasets/${state.activeDatasetId}/images/${encodeURIComponent(state.activeImageMeta.filename)}/annotations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_width: state.imageObj.width,
        image_height: state.imageObj.height,
        is_human: true,
        action: 'accepted',
        al_categories: tags.length > 0 ? tags : ['clear_positive'],
        boxes: clamped.map(b => ({
          ...b,
          needs_review: false
        }))
      })
    });

    // Flash green feedback
    elements.btnAccept.classList.add('flash-success');
    setTimeout(() => elements.btnAccept.classList.remove('flash-success'), 600);

    setDirty(false);
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
  const validationErrors = validateBoxesForSave(state.boxes);
  if (validationErrors.length > 0) {
    alert('Cannot save -- fix these OBBs first:\n\n' + validationErrors.join('\n'));
    return;
  }
  const tags = getSelectedPositiveTags();
  showSpinner('Saving human corrections...');
  try {
    const clamped = clampBoxesToImageBounds(state.boxes, state.imageObj.width, state.imageObj.height);
    await fetchJson(API_BASE + `api/datasets/${state.activeDatasetId}/images/${encodeURIComponent(state.activeImageMeta.filename)}/annotations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_width: state.imageObj.width,
        image_height: state.imageObj.height,
        is_human: true,
        action: 'human_corrected',
        al_categories: tags,
        boxes: clamped.map(b => ({
          ...b,
          model_source: b.model_source || 'HUMAN',
          confidence: 1.0,
          needs_review: false
        }))
      })
    });

    // Flash blue feedback
    elements.btnAcceptVerify.classList.add('flash-info');
    setTimeout(() => elements.btnAcceptVerify.classList.remove('flash-info'), 600);

    setDirty(false);
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
    await fetchJson(API_BASE + `api/datasets/${state.activeDatasetId}/images/${encodeURIComponent(state.activeImageMeta.filename)}/reject`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        failure_type: failureType,
        negative_category: negativeCategory,
        al_categories: [failureType, negativeCategory]
      })
    });

    elements.btnReject.classList.add('flash-danger');
    setTimeout(() => elements.btnReject.classList.remove('flash-danger'), 600);

    setDirty(false);
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
 * Skip: uncertain candidate, keep for later review. Marks the image's
 * status distinctly (not unlabeled/verified/rejected) without touching its
 * existing predictions/annotations, then advances to the next image.
 */
async function skipCurrentAnnotation() {
  if (!state.activeImageMeta) return;
  try {
    await fetch(API_BASE + `api/datasets/${state.activeDatasetId}/images/${encodeURIComponent(state.activeImageMeta.filename)}/skip`, {
      method: 'POST',
    });
  } catch (err) {
    console.error('Skip failed:', err);
  }
  await loadDatasetImages(state.activeDatasetId);
  if (state.activeImageIndex < state.images.length - 1) {
    selectImage(state.activeImageIndex + 1);
  }
}

/**
 * Toggle fixed test set membership for current image.
 */
async function toggleTestSetCurrent() {
  if (!state.activeImageMeta) return;
  showSpinner('Updating test set isolation...');
  try {
    const res = await fetch(API_BASE + `api/datasets/${state.activeDatasetId}/images/${encodeURIComponent(state.activeImageMeta.filename)}/test_set`, {
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
    const res = await fetch(API_BASE + `api/datasets/${state.activeDatasetId}/composition`);
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
    const res = await fetch(API_BASE + `api/datasets/${state.activeDatasetId}/export_test_set`, { method: 'POST' });
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
    const res = await fetch(API_BASE + `api/datasets/${state.activeDatasetId}/duplicates`);
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
    const res = await fetch(API_BASE + `api/datasets/${state.activeDatasetId}/next_difficult`);
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
    const res = await fetch(API_BASE + `api/datasets/${state.activeDatasetId}/active_learning${filterParam}`);
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
    const res = await fetch(API_BASE + `api/datasets/${state.activeDatasetId}/export_hard_cases?min_priority=60`, {
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
    const res = await fetch(API_BASE + `api/datasets/${state.activeDatasetId}/export`, { method: 'POST' });
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
    const res = await fetch(API_BASE + 'api/batch/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset_id: state.activeDatasetId,
        mode: elements.batchModeSelect.value,
        conf_threshold: 0.25,
        only_unlabeled: elements.batchUnlabeledOnly.checked,
        use_sam_refinement: elements.batchSamRefine.checked,
        enable_geometry_qa: elements.batchGeometryQa ? elements.batchGeometryQa.checked : true,
        qwen_gating: elements.batchQwenGating ? elements.batchQwenGating.value : null
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
    const res = await fetch(API_BASE + 'api/batch/status');
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
    const res = await fetch(API_BASE + `api/datasets/${state.activeDatasetId}/import`, {
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

const VALID_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.bmp'];
const IMPORT_UPLOAD_CONCURRENCY = 4;
let importQueueSeq = 0;

// Roboflow-style staged upload: files land in a thumbnail grid immediately
// (local preview only, nothing sent yet); "Import N Images" then uploads
// them with per-thumbnail progress, so the user can review/remove first.
function addFilesToStagingQueue(fileList) {
  const existingKeys = new Set(state.importQueue.map(item => `${item.file.name}:${item.file.size}`));
  const files = Array.from(fileList).filter(f =>
    VALID_IMAGE_EXTENSIONS.some(ext => f.name.toLowerCase().endsWith(ext)) &&
    !existingKeys.has(`${f.name}:${f.size}`)
  );
  if (!files.length) return;

  files.forEach(file => {
    const item = { id: ++importQueueSeq, file, url: URL.createObjectURL(file), status: 'pending', el: null };
    state.importQueue.push(item);
    renderImportThumb(item);
  });
  elements.importStaging.classList.remove('hidden');
  updateStagingHeader();
}

function renderImportThumb(item) {
  const card = document.createElement('div');
  card.className = 'import-thumb state-pending';

  const img = document.createElement('img');
  img.src = item.url;
  img.alt = item.file.name;
  card.appendChild(img);

  const badge = document.createElement('div');
  badge.className = 'thumb-status';
  card.appendChild(badge);

  const remove = document.createElement('button');
  remove.className = 'thumb-remove';
  remove.type = 'button';
  remove.title = 'Remove';
  remove.textContent = '×';
  remove.addEventListener('click', () => removeFromStagingQueue(item.id));
  card.appendChild(remove);

  const label = document.createElement('div');
  label.className = 'thumb-name';
  label.textContent = item.file.name;
  label.title = item.file.name;
  card.appendChild(label);

  item.el = card;
  elements.importThumbGrid.appendChild(card);
}

function removeFromStagingQueue(id) {
  const idx = state.importQueue.findIndex(item => item.id === id);
  if (idx === -1) return;
  const [item] = state.importQueue.splice(idx, 1);
  URL.revokeObjectURL(item.url);
  item.el?.remove();
  if (!state.importQueue.length) elements.importStaging.classList.add('hidden');
  updateStagingHeader();
}

function clearStagingQueue() {
  state.importQueue.forEach(item => URL.revokeObjectURL(item.url));
  state.importQueue = [];
  elements.importThumbGrid.innerHTML = '';
  elements.importStaging.classList.add('hidden');
  updateStagingHeader();
}

function updateStagingHeader() {
  const n = state.importQueue.length;
  elements.importStagingCount.textContent = `${n} image${n === 1 ? '' : 's'} ready`;
  elements.btnStartUpload.textContent = `Import ${n} Image${n === 1 ? '' : 's'}`;
  elements.btnStartUpload.disabled = n === 0 || state.importUploading;
}

function closeImportModal() {
  if (state.importUploading) return;
  elements.importModal.classList.add('hidden');
  clearStagingQueue();
}

async function uploadOneStagedFile(item) {
  item.status = 'uploading';
  item.el.className = 'import-thumb state-uploading';
  try {
    const formData = new FormData();
    formData.append('files', item.file, item.file.name);
    const res = await fetch(API_BASE + `api/datasets/${state.activeDatasetId}/import_upload`, {
      method: 'POST',
      body: formData
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (!data.imported_count) throw new Error('not imported');
    item.status = 'done';
    item.el.className = 'import-thumb state-done';
  } catch (err) {
    item.status = 'error';
    item.el.className = 'import-thumb state-error';
    item.el.title = `Failed: ${err.message}`;
  }
}

async function runWithConcurrency(items, worker, concurrency) {
  let next = 0;
  async function lane() {
    while (next < items.length) {
      const item = items[next++];
      await worker(item);
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, lane));
}

async function startStagedUpload() {
  const pending = state.importQueue.filter(item => item.status === 'pending' || item.status === 'error');
  if (!pending.length || !state.activeDatasetId) return;

  state.importUploading = true;
  elements.btnStartUpload.disabled = true;
  elements.btnClearStaging.disabled = true;
  elements.btnCancelImport.disabled = true;

  let done = 0;
  const updateProgress = () => {
    done++;
    elements.importStagingCount.textContent = `Uploading ${done} / ${pending.length}...`;
  };

  await runWithConcurrency(pending, async (item) => {
    await uploadOneStagedFile(item);
    updateProgress();
  }, IMPORT_UPLOAD_CONCURRENCY);

  const failed = state.importQueue.filter(item => item.status === 'error').length;
  const succeeded = state.importQueue.length - failed;
  elements.importStagingCount.textContent = failed
    ? `Imported ${succeeded}, ${failed} failed -- click Import to retry failed`
    : `Imported ${succeeded} image${succeeded === 1 ? '' : 's'}.`;

  state.importUploading = false;
  elements.btnClearStaging.disabled = false;
  elements.btnCancelImport.disabled = false;
  // Set button state directly (not via updateStagingHeader()) so the
  // "Imported N images." / "N failed" summary above stays visible instead
  // of being immediately clobbered back to "N images ready".
  if (failed) {
    elements.btnStartUpload.disabled = false;
    elements.btnStartUpload.textContent = 'Retry Failed';
  } else {
    const n = state.importQueue.length;
    elements.btnStartUpload.disabled = true;
    elements.btnStartUpload.textContent = `Import ${n} Image${n === 1 ? '' : 's'}`;
  }

  await loadDatasetImages(state.activeDatasetId);

  if (!failed) {
    setTimeout(() => closeImportModal(), 900);
  }
}

// --- KEYBOARD SHORTCUTS ---

function handleKeyDown(e) {
  // Ignore shortcuts when typing in an input
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;

  const ctrlOrCmd = e.ctrlKey || e.metaKey;
  if (ctrlOrCmd && (e.key === 'z' || e.key === 'Z')) {
    e.preventDefault();
    if (e.shiftKey) redo(); else undo();
    return;
  }
  if (ctrlOrCmd && (e.key === 'y' || e.key === 'Y')) {
    e.preventDefault();
    redo();
    return;
  }

  if (e.key === 'a' || e.key === 'A') {
    e.preventDefault();
    acceptCurrentAnnotation();
  } else if (e.key === 'r' || e.key === 'R') {
    e.preventDefault();
    rejectCurrentAnnotation();
  } else if (e.key === 'k' || e.key === 'K') {
    e.preventDefault();
    skipCurrentAnnotation();
  } else if (e.key === ' ' || e.key === 'Enter') {
    e.preventDefault();
    saveEditsCurrentAnnotation();
  } else if (e.key === 's' || e.key === 'S') {
    refineSelectedWithSAM();
  } else if (e.key === 'd' || e.key === 'D' || e.key === 'Delete' || e.key === 'Backspace') {
    // Backspace must be prevented here -- unhandled, most browsers treat it
    // as "navigate back" once focus isn't in an editable field.
    e.preventDefault();
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
