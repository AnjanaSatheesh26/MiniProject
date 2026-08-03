/**
 * content.js
 * ==========
 * Injected into every webpage by the Chrome extension.
 *
 * Responsibilities:
 *   1. Find all headline elements on the page (h1, h2, h3, article titles, etc.)
 *   2. Send headline text to inference engine (inference.js)
 *   3. Annotate headlines with:
 *      - Red border for clickbait
 *      - Highlighted words with explanation
 *      - Clickbait badge showing confidence
 *   4. Handle dynamic pages (MutationObserver for SPA sites)
 *
 * PRIVACY:
 *   - Headline text never leaves the browser
 *   - No content is logged or stored
 *   - Only visual DOM annotations are made (all reversible)
 *   - MutationObserver watches DOM only for new headlines, not content
 */

'use strict';

// ================================================================
// CONFIGURATION
// ================================================================
const CONFIG = {
  // Minimum confidence threshold to mark as clickbait
  CLICKBAIT_THRESHOLD: 0.65,

  // CSS selectors for headline elements to analyze
  HEADLINE_SELECTORS: [
    'h1', 'h2', 'h3',
    'article h4', 'article h5',
    '.headline', '.title', '.article-title',
    '[data-testid*="headline"]', '[data-testid*="title"]',
    '.card-title', '.post-title', '.story-title',
    'a[href] h1', 'a[href] h2', 'a[href] h3',
  ].join(', '),

  // Maximum text length to send to inference (characters)
  MAX_HEADLINE_LENGTH: 512,

  // Minimum text length to analyze (skip very short text)
  MIN_HEADLINE_LENGTH: 10,

  // Visual styling for annotations
  CLICKBAIT_BORDER_COLOR:  '#e53e3e',
  SAFE_BORDER_COLOR:       '#38a169',
  HIGHLIGHT_COLOR_HIGH:    'rgba(229, 62, 62, 0.3)',   // High-score word highlight
  HIGHLIGHT_COLOR_MED:     'rgba(255, 165, 0, 0.2)',   // Medium-score word highlight

  // Delay before re-analyzing after DOM changes (ms)
  DEBOUNCE_DELAY: 1000,
};

// ================================================================
// STATE
// ================================================================
let inferenceReady    = false;
let extensionEnabled  = true;
let analyzedElements  = new WeakSet();  // Track which elements we've annotated
let debounceTimer     = null;

// ================================================================
// INITIALIZATION
// ================================================================

/**
 * Initialize: load inference engine, check extension state, analyze page.
 */
async function init() {
  // Check if extension is enabled
  try {
    const response = await chrome.runtime.sendMessage({ type: 'GET_STATS' });
    extensionEnabled = response?.data?.enabled ?? true;
  } catch (e) {
    console.warn('[ClickbaitDetector] Could not reach background script:', e.message);
  }

  if (!extensionEnabled) return;

  // Load onnxruntime-web from CDN
  // This is the only external resource loaded — purely a JS runtime library,
  // not a data or analytics service. The model runs entirely locally.
  await loadOnnxRuntime();

  // Load and initialize inference engine
  await loadInferenceScript();

  // Analyze existing page content
  await analyzeVisibleHeadlines();

  // Watch for new headlines (SPAs, infinite scroll, etc.)
  startMutationObserver();

  console.log('[ClickbaitDetector] Initialized on:', window.location.hostname);
}

/**
 * Dynamically load onnxruntime-web from CDN.
 * Needed because MV3 content scripts can't import ES modules from external URLs.
 */
function loadOnnxRuntime() {
  return new Promise((resolve, reject) => {
    if (typeof ort !== 'undefined') {
      resolve();
      return;
    }

    const script = document.createElement('script');
    // Use jsdelivr CDN (fast, reliable, no tracking)
    script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.0/dist/ort.min.js';
    script.onload = () => {
      console.log('[ClickbaitDetector] onnxruntime-web loaded');
      resolve();
    };
    script.onerror = (e) => {
      console.error('[ClickbaitDetector] Failed to load onnxruntime-web:', e);
      // Fail gracefully — don't crash the page
      reject(new Error('onnxruntime-web load failed'));
    };
    document.head.appendChild(script);
  });
}

/**
 * Load and initialize the local inference script (inference.js).
 * inference.js is bundled with the extension and runs locally.
 */
function loadInferenceScript() {
  return new Promise((resolve, reject) => {
    if (window.CLICKBAIT_INFERENCE) {
      // Already loaded
      inferenceReady = true;
      resolve();
      return;
    }

    const script = document.createElement('script');
    script.src = chrome.runtime.getURL('utils/inference.js');
    script.onload = async () => {
      try {
        await window.CLICKBAIT_INFERENCE.initialize();
        inferenceReady = true;
        console.log('[ClickbaitDetector] Inference engine ready');
        resolve();
      } catch (e) {
        console.error('[ClickbaitDetector] Inference init failed:', e);
        reject(e);
      }
    };
    script.onerror = reject;
    document.head.appendChild(script);
  });
}

// ================================================================
// HEADLINE DETECTION
// ================================================================

/**
 * Find all unanalyzed headline elements on the current page.
 * @returns {Element[]} Array of headline DOM elements
 */
function findHeadlineElements() {
  const elements = document.querySelectorAll(CONFIG.HEADLINE_SELECTORS);
  const unanalyzed = [];

  elements.forEach(el => {
    // Skip already-analyzed elements
    if (analyzedElements.has(el)) return;

    // Skip our own injected elements
    if (el.dataset.clickbaitAnnotated) return;

    const text = getCleanText(el);
    if (text.length >= CONFIG.MIN_HEADLINE_LENGTH) {
      unanalyzed.push(el);
    }
  });

  return unanalyzed;
}

/**
 * Extract clean text from a DOM element, stripping our annotations.
 * @param {Element} el - DOM element
 * @returns {string} Cleaned text content
 */
function getCleanText(el) {
  // Use innerText but remove our annotation spans
  const clone = el.cloneNode(true);
  clone.querySelectorAll('[data-clickbait-annotation]').forEach(n => n.remove());
  const text = (clone.innerText || clone.textContent || '').trim();
  return text.slice(0, CONFIG.MAX_HEADLINE_LENGTH);
}

// ================================================================
// ANALYSIS PIPELINE
// ================================================================

/**
 * Analyze all currently-visible headline elements.
 * Runs in batches to avoid blocking the main thread.
 */
async function analyzeVisibleHeadlines() {
  if (!inferenceReady || !extensionEnabled) return;

  const elements = findHeadlineElements();
  if (elements.length === 0) return;

  console.log(`[ClickbaitDetector] Analyzing ${elements.length} headlines`);

  // Mark elements as analyzed immediately to prevent re-analysis
  elements.forEach(el => analyzedElements.add(el));

  // Process in small batches to keep UI responsive
  const BATCH_SIZE = 5;
  let totalClickbait = 0;

  for (let i = 0; i < elements.length; i += BATCH_SIZE) {
    const batch = elements.slice(i, i + BATCH_SIZE);
    const headlines = batch.map(el => getCleanText(el));

    try {
      const results = await window.CLICKBAIT_INFERENCE.analyzeHeadlines(headlines);

      results.forEach((result, idx) => {
        if (result.error) return;

        const element = batch[idx];
        if (result.prediction === 1 && result.confidence >= CONFIG.CLICKBAIT_THRESHOLD) {
          annotateClickbait(element, result);
          totalClickbait++;
        } else {
          annotateSafe(element, result);
        }
      });

    } catch (error) {
      console.error('[ClickbaitDetector] Batch inference failed:', error);
    }

    // Yield to browser between batches
    await new Promise(resolve => setTimeout(resolve, 10));
  }

  // Update stats in background
  try {
    chrome.runtime.sendMessage({
      type:      'UPDATE_STATS',
      analyzed:  elements.length,
      clickbait: totalClickbait,
    });
  } catch (e) { /* Ignore if background is unavailable */ }
}

// ================================================================
// DOM ANNOTATION
// ================================================================

/**
 * Annotate a headline element as clickbait.
 * Adds:
 *   - Red left border
 *   - Highlighted clickbait words
 *   - Confidence badge
 *   - Tooltip on hover
 *
 * @param {Element} element - The headline DOM element
 * @param {object}  result  - Inference result from inference.js
 */
function annotateClickbait(element, result) {
  // Mark element to prevent re-processing
  element.dataset.clickbaitAnnotated = 'true';
  element.dataset.clickbaitResult    = JSON.stringify({
    prediction:  result.prediction,
    confidence:  result.confidence,
    label:       result.label,
    lime_words:  result.lime_words,
  });

  // Red left border to flag clickbait
  element.style.borderLeft    = `4px solid ${CONFIG.CLICKBAIT_BORDER_COLOR}`;
  element.style.paddingLeft   = '8px';
  element.style.position      = 'relative';
  element.style.marginLeft    = '4px';
  element.style.transition    = 'all 0.2s ease';

  // Apply word-level highlighting
  if (result.lime_words && result.lime_words.length > 0) {
    highlightWords(element, result.lime_words);
  }

  // Add confidence badge
  addConfidenceBadge(element, result.confidence, 'clickbait');

  // Store full result for popup access
  element.dataset.clickbaitFullResult = JSON.stringify(result);
}

/**
 * Annotate a headline as non-clickbait (subtle green indicator).
 */
function annotateSafe(element, result) {
  element.dataset.clickbaitAnnotated = 'true';
  element.dataset.clickbaitResult    = JSON.stringify({
    prediction: result.prediction,
    confidence: result.confidence,
    label:      result.label,
  });

  // Very subtle indicator — don't clutter safe headlines
  element.style.borderLeft  = `2px solid ${CONFIG.SAFE_BORDER_COLOR}`;
  element.style.paddingLeft = '6px';
  element.style.marginLeft  = '4px';
}

/**
 * Highlight clickbait words within a headline element.
 * Wraps high-scoring words in colored <span> elements.
 *
 * @param {Element}  element   - Headline DOM element
 * @param {Array}    limeWords - Array of {word, score} objects
 */
function highlightWords(element, limeWords) {
  const originalHTML = element.innerHTML;

  // Build word → score map (case-insensitive)
  const wordScores = {};
  limeWords.forEach(({ word, score }) => {
    if (score > 0.3) {  // Only highlight meaningful contributors
      wordScores[word.toLowerCase()] = score;
    }
  });

  if (Object.keys(wordScores).length === 0) return;

  // Build a regex pattern matching any of the high-score words
  const pattern = Object.keys(wordScores)
    .map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))  // Escape regex chars
    .join('|');

  if (!pattern) return;

  const regex = new RegExp(`\\b(${pattern})\\b`, 'gi');

  // Walk text nodes and replace matching words with highlighted spans
  // We do this on text nodes to avoid corrupting existing HTML
  walkTextNodes(element, (textNode) => {
    const text = textNode.textContent;
    if (!regex.test(text)) return;

    regex.lastIndex = 0;
    const fragment = document.createDocumentFragment();
    let lastIndex  = 0;
    let match;

    while ((match = regex.exec(text)) !== null) {
      // Text before this match
      if (match.index > lastIndex) {
        fragment.appendChild(document.createTextNode(text.slice(lastIndex, match.index)));
      }

      const word  = match[0];
      const score = wordScores[word.toLowerCase()] || 0;
      const color = score > 0.6 ? CONFIG.HIGHLIGHT_COLOR_HIGH : CONFIG.HIGHLIGHT_COLOR_MED;

      const span = document.createElement('span');
      span.dataset.clickbaitAnnotation = 'true';
      span.style.backgroundColor = color;
      span.style.borderRadius    = '2px';
      span.style.padding         = '1px 2px';
      span.style.cursor          = 'help';
      span.title = `Clickbait signal: "${word}" (score: ${(score * 100).toFixed(0)}%)`;
      span.textContent = word;

      fragment.appendChild(span);
      lastIndex = match.index + word.length;
    }

    // Remaining text after last match
    if (lastIndex < text.length) {
      fragment.appendChild(document.createTextNode(text.slice(lastIndex)));
    }

    textNode.parentNode.replaceChild(fragment, textNode);
  });
}

/**
 * Walk all text nodes within an element.
 * Avoids modifying elements that are already annotation spans.
 */
function walkTextNodes(element, callback) {
  const walker = document.createTreeWalker(
    element,
    NodeFilter.SHOW_TEXT,
    {
      acceptNode: (node) => {
        // Skip text inside our own annotation elements
        if (node.parentElement?.dataset.clickbaitAnnotation) {
          return NodeFilter.FILTER_REJECT;
        }
        return NodeFilter.FILTER_ACCEPT;
      },
    }
  );

  const textNodes = [];
  let node;
  while ((node = walker.nextNode())) {
    textNodes.push(node);
  }

  // Modify after collection to avoid invalidating the walker
  textNodes.forEach(callback);
}

/**
 * Add a small confidence badge near the headline.
 *
 * @param {Element} element    - Headline element
 * @param {number}  confidence - P(clickbait) [0, 1]
 * @param {string}  type       - 'clickbait' or 'non-clickbait'
 */
function addConfidenceBadge(element, confidence, type) {
  // Remove any existing badge
  const existingBadge = element.querySelector('[data-clickbait-badge]');
  if (existingBadge) existingBadge.remove();

  const badge = document.createElement('span');
  badge.dataset.clickbaitBadge = 'true';

  const pct = Math.round(confidence * 100);
  badge.textContent = `⚠ ${pct}% clickbait`;

  // Style the badge
  Object.assign(badge.style, {
    display:         'inline-block',
    marginLeft:      '8px',
    padding:         '2px 6px',
    borderRadius:    '12px',
    fontSize:        '11px',
    fontWeight:      'bold',
    color:           '#fff',
    backgroundColor: confidence > 0.8 ? '#c53030' : '#dd6b20',
    verticalAlign:   'middle',
    cursor:          'help',
    lineHeight:      '1.4',
  });

  badge.title = `This headline has a ${pct}% probability of being clickbait according to the local AI model.`;

  element.appendChild(badge);
}

// ================================================================
// MUTATION OBSERVER — handles SPAs and dynamic content
// ================================================================

/**
 * Start watching the DOM for new headlines.
 * Handles infinite scroll, React/Vue SPAs, and AJAX-loaded content.
 */
function startMutationObserver() {
  const observer = new MutationObserver((mutations) => {
    // Check if any added nodes contain headline elements
    let hasNewContent = false;

    for (const mutation of mutations) {
      if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
        for (const node of mutation.addedNodes) {
          if (node.nodeType === Node.ELEMENT_NODE) {
            // Check if the added node itself is a headline or contains headlines
            if (
              node.matches && (
                node.matches(CONFIG.HEADLINE_SELECTORS) ||
                node.querySelector(CONFIG.HEADLINE_SELECTORS)
              )
            ) {
              hasNewContent = true;
              break;
            }
          }
        }
      }
      if (hasNewContent) break;
    }

    if (hasNewContent) {
      // Debounce to avoid excessive analysis on rapid DOM changes
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(analyzeVisibleHeadlines, CONFIG.DEBOUNCE_DELAY);
    }
  });

  observer.observe(document.body, {
    childList: true,
    subtree:   true,
  });
}

// ================================================================
// MESSAGE LISTENER
// Handles messages from background.js and popup.js
// ================================================================
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'EXTENSION_STATE_CHANGED') {
    extensionEnabled = message.enabled;
    if (extensionEnabled) {
      analyzeVisibleHeadlines();
    }
    sendResponse({ success: true });
  }

  if (message.type === 'GET_PAGE_RESULTS') {
    // Collect all annotated headlines for the popup
    const results = [];
    document.querySelectorAll('[data-clickbait-annotated]').forEach(el => {
      try {
        const data = JSON.parse(el.dataset.clickbaitResult || '{}');
        results.push({
          headline:   getCleanText(el),
          prediction: data.prediction,
          confidence: data.confidence,
          label:      data.label,
        });
      } catch (e) {}
    });
    sendResponse({ success: true, results });
  }

  return true;
});

// ================================================================
// START
// ================================================================

// Wait for DOM to be ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
