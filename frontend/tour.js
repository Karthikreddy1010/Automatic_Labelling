/**
 * frontend/tour.js -- Guided product tour for the PoleAnnotator workspace.
 *
 * Self-contained: it reads nothing from app.js's state and mutates no
 * annotation data, so it can run at any point without disturbing work in
 * progress. app.js only needs to (a) suspend its keyboard shortcuts while a
 * tour is running and (b) call PoleTour.start() from the Guide button.
 *
 * Each step points at a live element by selector. A step whose element is
 * missing (feature hidden, markup changed) is skipped rather than breaking
 * the run, so the tour degrades instead of dead-ending.
 */
(function () {
  'use strict';

  const SEEN_KEY = 'poleannotator.tour.completed.v1';
  const GAP = 14;        // px between the spotlight and the tooltip
  const PAD = 8;         // px of breathing room around the highlighted element
  const MARGIN = 12;     // px minimum distance from the viewport edge

  /**
   * The tour script. `target` is a CSS selector (first match wins); omit it
   * for a centred card with no spotlight. `prefer` nudges which side the
   * tooltip is placed on when there is room for more than one.
   */
  const STEPS = [
    {
      title: 'Welcome to PoleAnnotator',
      body: 'A two-minute tour of the labeling workspace: where images come in, how the AI pipeline proposes oriented boxes, and how you review them. Use <kbd>&larr;</kbd> / <kbd>&rarr;</kbd> to move, <kbd>Esc</kbd> to leave at any point.',
    },
    {
      target: '.dataset-pill',
      prefer: 'bottom',
      title: '1. Your dataset',
      body: 'Switch datasets here. <strong>+</strong> creates a new one, and the upload icon opens the importer -- drag in a folder of images, or paste a local path.',
    },
    {
      target: '#btn-ai-label',
      prefer: 'bottom',
      title: '2. Auto-label one image',
      body: '<strong>AI Label</strong> runs the production pipeline on the current image: Grounding DINO finds candidates, SAM refines their masks, geometry QA and Qwen verify them, and the decision engine marks each one ACCEPT, REVIEW or REJECT. <strong>Run All</strong> instead runs every model side by side for comparison.',
    },
    {
      target: '#btn-batch-modal',
      prefer: 'bottom',
      title: '3. Auto-label the whole dataset',
      body: 'The <strong>Batch Engine</strong> runs that same pipeline across every image, with a progress tracker you can pause, resume or cancel. This is the usual starting point on a fresh dataset.',
    },
    {
      target: '.filter-pills',
      prefer: 'right',
      title: '4. Pick what to work on',
      body: 'Narrow the image list by status. <strong>AI Suggested</strong> is the review queue -- images the pipeline has labelled but no human has confirmed yet.',
    },
    {
      target: '#dataset-stats-card',
      prefer: 'right',
      title: '5. Where the dataset stands',
      body: 'Verified, needs-review and unlabeled counts for the whole dataset, refreshed after every action you take.',
    },
    {
      target: '#image-gallery',
      prefer: 'right',
      title: '6. The image list',
      body: 'Click any image to open it. The tag on the right is its status; <span class="tour-chip">TEST</span> marks the fixed benchmark set and <span class="tour-chip">DUP</span> a near-duplicate.',
    },
    {
      target: '#gallery-tools',
      prefer: 'right',
      title: '7. Taking images back out',
      body: 'Imported the wrong folder? <strong>Undo upload</strong> removes everything the last import added, leaving earlier images alone, so you can import a different batch. <strong>Select</strong> turns the list into a checklist for removing particular images, hovering a row gives it an &times;, and <strong>Clear all</strong> empties the dataset. Removing an image also deletes its labels, so each of these asks first.',
    },
    {
      target: '#viewport',
      prefer: 'left',
      title: '8. The canvas',
      body: 'Click a box to select it, drag its corners to reshape it, drag the handle above it to rotate, and drag the body to move it. Scroll to zoom, drag empty space to pan, <kbd>F</kbd> to fit.',
    },
    {
      target: '#tool-draw',
      prefer: 'bottom',
      title: '9. Drawing a box the AI missed',
      body: 'Switch to the draw tool (<kbd>W</kbd>) and drag out a new oriented box; <kbd>V</kbd> or <kbd>Esc</kbd> returns to select mode. <kbd>S</kbd> refines the selected box with SAM, <kbd>D</kbd> deletes it, and <kbd>Ctrl</kbd>+<kbd>Z</kbd> undoes.',
    },
    {
      target: '#model-toggles-bar',
      prefer: 'bottom',
      title: '10. Compare the models',
      body: 'After a <strong>Run All</strong>, each model draws in its own colour. Toggle the layers to see exactly where YOLO, DINO, SAM and the OBB recovery stage disagree.',
    },
    {
      target: '.labels-list-card',
      prefer: 'left',
      title: '11. Every box on this image',
      body: 'One row per box, with the pipeline decision on each. The chips filter the list down to just the boxes needing attention -- review flags, model disagreement, or geometry warnings.',
    },
    {
      target: '.inspector-card',
      prefer: 'left',
      title: '12. Why the AI proposed this',
      body: 'Select a box and this panel shows its full evidence trail: confidence, tilt and area, which models contributed, the geometry and Qwen verdicts, and the active-learning tags you can attach as you review.',
    },
    {
      target: '.action-btn-cluster',
      prefer: 'top',
      title: '13. Reviewing: the four verdicts',
      body: '<strong>Accept</strong> (<kbd>A</kbd>) takes the AI boxes as ground truth. <strong>Save Edits</strong> (<kbd>Space</kbd>) stores your corrections. <strong>Reject</strong> (<kbd>R</kbd>) records a false positive as a hard negative. <strong>Skip</strong> (<kbd>K</kbd>) parks an uncertain one. All four save, then move you <em>forward</em> to the next image in the list automatically.',
    },
    {
      target: '#image-pagination-text',
      prefer: 'top',
      title: '14. Moving by hand',
      body: 'Your position in the current list, with <strong>Prev</strong> / <strong>Next</strong> (<kbd>P</kbd> / <kbd>N</kbd>, or the arrow keys) either side of it. Unsaved edits are flagged here before you navigate away.',
    },
    {
      target: '.active-learning-card',
      prefer: 'left',
      title: '15. Let active learning choose',
      body: 'Rather than working top to bottom, <strong>Review Next</strong> (<kbd>Q</kbd>) jumps to the highest-value image still pending -- the ones where the models disagreed or the geometry looked wrong. <strong>Export Hard</strong> pulls just those out as a training subset.',
    },
    {
      target: '.coverage-card',
      prefer: 'left',
      title: '16. Keeping the dataset balanced',
      body: 'Positives against hard negatives, false positives, missed poles and redundant near-duplicates -- plus warnings when the mix starts to skew. Keep a fixed test set here so your benchmark never drifts.',
    },
    {
      target: '#btn-export-dataset',
      prefer: 'bottom',
      title: '17. Export for training',
      body: 'Writes the verified annotations out in YOLO-OBB format, ready to train on.',
    },
    {
      target: '#btn-shortcuts',
      prefer: 'bottom',
      title: 'That is the tour',
      body: '<kbd>?</kbd> opens the full shortcut list, and <kbd>G</kbd> replays this tour whenever you want it.',
    },
  ];

  let steps = [];        // resolved steps for this run
  let index = 0;
  let active = false;
  let nodes = null;      // { overlay, spotlight, card, title, body, counter, progress, back, next, skip }

  // ---------------------------------------------------------------- DOM

  function build() {
    const overlay = document.createElement('div');
    overlay.className = 'tour-overlay';
    overlay.innerHTML = `
      <div class="tour-spotlight" aria-hidden="true"></div>
      <div class="tour-card" role="dialog" aria-modal="true" aria-labelledby="tour-title">
        <div class="tour-card-head">
          <span class="tour-counter"></span>
          <button type="button" class="tour-skip" title="End tour (Esc)">Skip tour</button>
        </div>
        <h3 class="tour-title" id="tour-title"></h3>
        <div class="tour-body"></div>
        <div class="tour-foot">
          <div class="tour-progress"><div class="tour-progress-fill"></div></div>
          <div class="tour-nav">
            <button type="button" class="tour-btn tour-back">Back</button>
            <button type="button" class="tour-btn tour-btn-primary tour-next">Next</button>
          </div>
        </div>
      </div>`;
    document.body.appendChild(overlay);

    nodes = {
      overlay,
      spotlight: overlay.querySelector('.tour-spotlight'),
      card: overlay.querySelector('.tour-card'),
      title: overlay.querySelector('.tour-title'),
      body: overlay.querySelector('.tour-body'),
      counter: overlay.querySelector('.tour-counter'),
      progress: overlay.querySelector('.tour-progress-fill'),
      back: overlay.querySelector('.tour-back'),
      next: overlay.querySelector('.tour-next'),
      skip: overlay.querySelector('.tour-skip'),
    };

    nodes.back.addEventListener('click', () => go(index - 1));
    nodes.next.addEventListener('click', () => go(index + 1));
    nodes.skip.addEventListener('click', () => stop(false));
    // Clicking the backdrop -- including over the highlighted element, which
    // the tour makes click-through -- advances. Clicking the card must not.
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) go(index + 1);
    });
    nodes.card.addEventListener('click', (e) => e.stopPropagation());
    return nodes;
  }

  // ------------------------------------------------------------ placing

  /** Centre the card and hide the spotlight -- for steps with no target. */
  function placeCentred() {
    nodes.spotlight.classList.add('tour-hidden');
    nodes.card.classList.add('tour-card-centred');
    nodes.card.style.left = '';
    nodes.card.style.top = '';
  }

  /**
   * Put the spotlight over `rect` and the card on whichever side has room,
   * trying the step's preferred side first. Falls back to centring the card
   * when the element fills the viewport.
   */
  function placeAt(rect, prefer) {
    const vw = window.innerWidth;
    const vh = window.innerHeight;

    // Clamped: an element flush against an edge would otherwise push its
    // padded spotlight off-screen, and the ring would lose a side.
    const sLeft = Math.max(2, rect.left - PAD);
    const sTop = Math.max(2, rect.top - PAD);
    nodes.spotlight.classList.remove('tour-hidden');
    nodes.spotlight.style.left = `${sLeft}px`;
    nodes.spotlight.style.top = `${sTop}px`;
    nodes.spotlight.style.width = `${Math.min(rect.right + PAD, vw - 2) - sLeft}px`;
    nodes.spotlight.style.height = `${Math.min(rect.bottom + PAD, vh - 2) - sTop}px`;

    nodes.card.classList.remove('tour-card-centred');
    const cw = nodes.card.offsetWidth;
    const ch = nodes.card.offsetHeight;

    const room = {
      bottom: vh - rect.bottom - GAP - MARGIN,
      top: rect.top - GAP - MARGIN,
      right: vw - rect.right - GAP - MARGIN,
      left: rect.left - GAP - MARGIN,
    };
    const order = [prefer, 'bottom', 'top', 'right', 'left'].filter(Boolean);
    const side = order.find(s => room[s] >= (s === 'top' || s === 'bottom' ? ch : cw));

    if (!side) { placeCentred(); return; }

    let left;
    let top;
    if (side === 'bottom' || side === 'top') {
      left = rect.left + rect.width / 2 - cw / 2;
      top = side === 'bottom' ? rect.bottom + GAP : rect.top - GAP - ch;
    } else {
      left = side === 'right' ? rect.right + GAP : rect.left - GAP - cw;
      top = rect.top + rect.height / 2 - ch / 2;
    }
    nodes.card.style.left = `${clamp(left, MARGIN, vw - cw - MARGIN)}px`;
    nodes.card.style.top = `${clamp(top, MARGIN, vh - ch - MARGIN)}px`;
  }

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(v, hi));
  }

  /** Scroll a target into view inside its own scroll container, if needed. */
  function reveal(el) {
    const r = el.getBoundingClientRect();
    const offscreen = r.top < 0 || r.bottom > window.innerHeight ||
                      r.left < 0 || r.right > window.innerWidth;
    if (offscreen) el.scrollIntoView({ block: 'center', inline: 'nearest' });
  }

  // ------------------------------------------------------------ stepping

  function show() {
    const step = steps[index];
    nodes.title.innerHTML = step.title;
    nodes.body.innerHTML = step.body;
    nodes.counter.textContent = `Step ${index + 1} of ${steps.length}`;
    nodes.back.disabled = index === 0;
    nodes.next.textContent = index === steps.length - 1 ? 'Done' : 'Next';
    nodes.progress.style.width = `${((index + 1) / steps.length) * 100}%`;

    if (lastHighlighted) lastHighlighted.classList.remove('tour-target');

    const el = step.target ? document.querySelector(step.target) : null;
    if (!el) {
      lastHighlighted = null;
      placeCentred();
      return;
    }
    reveal(el);
    el.classList.add('tour-target');
    lastHighlighted = el;
    // Measure after the scroll has settled, otherwise the spotlight lands
    // on where the element used to be.
    requestAnimationFrame(() => {
      if (!active || steps[index] !== step) return;
      placeAt(el.getBoundingClientRect(), step.prefer);
    });
  }

  let lastHighlighted = null;

  function go(to) {
    if (!active) return;
    if (to < 0) return;
    if (to >= steps.length) { stop(true); return; }
    index = to;
    show();
  }

  function onKey(e) {
    if (!active) return;
    // The tour owns the keyboard while it is up, so app.js's single-letter
    // shortcuts cannot fire behind it.
    e.stopPropagation();
    if (e.key === 'Escape') { e.preventDefault(); stop(false); }
    else if (e.key === 'ArrowRight' || e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(index + 1); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); go(index - 1); }
  }

  function onReflow() {
    if (!active) return;
    const step = steps[index];
    const el = step.target ? document.querySelector(step.target) : null;
    if (el) placeAt(el.getBoundingClientRect(), step.prefer);
    else placeCentred();
  }

  // -------------------------------------------------------------- public

  function start(from = 0) {
    if (active) return;
    // Drop steps whose target is not in this build of the page.
    steps = STEPS.filter(s => !s.target || document.querySelector(s.target));
    if (steps.length === 0) return;
    active = true;
    index = clamp(from, 0, steps.length - 1);
    build();
    document.body.classList.add('tour-running');
    // Capture phase, so the tour sees keys before app.js's window handler.
    window.addEventListener('keydown', onKey, true);
    window.addEventListener('resize', onReflow);
    window.addEventListener('scroll', onReflow, true);
    show();
  }

  function stop(completed) {
    if (!active) return;
    active = false;
    window.removeEventListener('keydown', onKey, true);
    window.removeEventListener('resize', onReflow);
    window.removeEventListener('scroll', onReflow, true);
    if (lastHighlighted) lastHighlighted.classList.remove('tour-target');
    lastHighlighted = null;
    document.body.classList.remove('tour-running');
    if (nodes) { nodes.overlay.remove(); nodes = null; }
    if (completed) markSeen();
  }

  function markSeen() {
    try { localStorage.setItem(SEEN_KEY, '1'); } catch (err) { /* private mode */ }
  }

  function hasSeen() {
    try { return localStorage.getItem(SEEN_KEY) === '1'; } catch (err) { return true; }
  }

  /** Run once for a first-time visitor; a no-op afterwards. */
  function maybeAutoStart() {
    if (hasSeen()) return;
    // Mark it immediately: if the tour is dismissed by closing the tab it
    // still should not reappear on every load.
    markSeen();
    setTimeout(() => start(0), 900);
  }

  window.PoleTour = {
    start,
    stop: () => stop(false),
    maybeAutoStart,
    isActive: () => active,
  };
})();
