/**
 * popup.js
 * ========
 * Controls the extension popup UI.
 *
 * Responsibilities:
 *   - Load and display stats from chrome.storage
 *   - Fetch current page headline analysis results from content.js
 *   - Render result cards for each headline
 *   - Show word-level explanation panel when a headline is clicked
 *   - Handle enable/disable toggle
 *   - Handle rescan button
 *
 * PRIVACY NOTE:
 *   popup.js only reads data from chrome.storage and the active tab's
 *   content script. No data is sent anywhere externally.
 */

'use strict';

// ================================================================
// DOM REFERENCES
// ================================================================
const enabledToggle       = document.getElementById('enabledToggle');
const statTotal           = document.getElementById('statTotal');
const statClickbait       = document.getElementById('statClickbait');
const statSafe            = document.getElementById('statSafe');
const statRate            = document.getElementById('statRate');
const loadingState        = document.getElementById('loadingState');
const emptyState          = document.getElementById('emptyState');
const resultsList         = document.getElementById('resultsList');
const rescanBtn           = document.getElementById('rescanBtn');
const explanationPanel    = document.getElementById('explanationPanel');
const closeExplanation    = document.getElementById('closeExplanation');
const explanationHeadline = document.getElementById('explanationHeadline');
const explanationWords    = document.getElementById('explanationWords');


// ================================================================
// INITIALIZATION
// ================================================================

document.addEventListener('DOMContentLoaded', async () => {
  await loadStoredStats();
  await loadPageResults();
  bindEvents();
});


/**
 * Load global stats from chrome.storage and update the stats bar.
 */
async function loadStoredStats() {
  try {
    const data = await chrome.storage.local.get(['totalAnalyzed', 'clickbaitFound', 'enabled']);

    const total    = data.totalAnalyzed  || 0;
    const clickbait = data.clickbaitFound || 0;
    const safe     = total - clickbait;
    const rate     = total > 0 ? `${Math.round((clickbait / total) * 100)}%` : '—';

    statTotal.textContent     = total;
    statClickbait.textContent = clickbait;
    statSafe.textContent      = safe;
    statRate.textContent      = rate;

    // Set toggle state
    enabledToggle.checked = data.enabled !== false;

  } catch (error) {
    console.error('[Popup] Failed to load stats:', error);
  }
}


/**
 * Request current page headline results from content.js.
 */
async function loadPageResults() {
  showLoading(true);

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    if (!tab?.id) {
      showEmptyState('Could not access active tab.');
      return;
    }

    // Send message to content.js requesting current page results
    let response;
    try {
      response = await chrome.tabs.sendMessage(tab.id, { type: 'GET_PAGE_RESULTS' });
    } catch (e) {
      // Content script may not be loaded (e.g., on chrome:// pages)
      showEmptyState('Extension not active on this page.\nTry a news website.');
      return;
    }

    showLoading(false);

    if (!response?.success || !response.results || response.results.length === 0) {
      showEmptyState();
      return;
    }

    renderResults(response.results);

  } catch (error) {
    console.error('[Popup] Failed to load page results:', error);
    showEmptyState('An error occurred. Try reloading the page.');
  }
}


// ================================================================
// RENDERING
// ================================================================

/**
 * Render the list of analyzed headlines.
 * @param {Array} results - Array of {headline, prediction, confidence, label, lime_words}
 */
function renderResults(results) {
  resultsList.innerHTML = '';

  // Sort: clickbait first (most confident first), then safe
  const sorted = [...results].sort((a, b) => {
    if (a.prediction !== b.prediction) return b.prediction - a.prediction;
    return b.confidence - a.confidence;
  });

  sorted.forEach((result, idx) => {
    const card = createResultCard(result, idx);
    resultsList.appendChild(card);
  });

  resultsList.classList.remove('hidden');
  emptyState.classList.add('hidden');
}


/**
 * Create a single result card element.
 * @param {object} result - Single headline analysis result
 * @param {number} idx    - Index for animation staggering
 * @returns {HTMLElement} The card element
 */
function createResultCard(result, idx) {
  const isClickbait = result.prediction === 1 || result.label === 'clickbait';
  const confidence  = typeof result.confidence === 'number' ? result.confidence : 0;
  const pct         = Math.round(confidence * 100);

  const card = document.createElement('div');
  card.className = `result-item ${isClickbait ? 'clickbait' : 'safe'}`;
  card.style.animationDelay = `${idx * 50}ms`;

  // Truncate long headlines for display
  const displayHeadline = truncate(result.headline || 'Unknown headline', 120);

  card.innerHTML = `
    <div class="result-headline">${escapeHtml(displayHeadline)}</div>
    <div class="result-meta">
      <span class="result-label ${isClickbait ? 'label-clickbait' : 'label-safe'}">
        ${isClickbait ? '⚠ Clickbait' : '✓ Safe'}
      </span>
      <span class="result-confidence">${pct}% confident</span>
      ${isClickbait ? '<span class="result-explain-hint">→ Explain</span>' : ''}
    </div>
    <div class="confidence-bar">
      <div
        class="confidence-fill ${isClickbait ? 'fill-clickbait' : 'fill-safe'}"
        style="width: ${pct}%"
      ></div>
    </div>
  `;

  // Click handler: show explanation panel for clickbait items
  if (isClickbait) {
    card.addEventListener('click', () => {
      showExplanation(result);
    });
  }

  return card;
}


/**
 * Show the word attribution explanation panel for a headline.
 * @param {object} result - Full inference result with lime_words
 */
function showExplanation(result) {
  const headline  = result.headline || '';
  const limeWords = result.lime_words || [];

  // Display headline text
  explanationHeadline.textContent = headline;

  // Render word attribution bars
  explanationWords.innerHTML = '';

  if (limeWords.length === 0) {
    explanationWords.innerHTML = '<p class="muted" style="padding:8px">No word attribution data available.</p>';
  } else {
    // Sort by absolute score
    const sorted = [...limeWords].sort((a, b) => Math.abs(b.score) - Math.abs(a.score));
    const top    = sorted.slice(0, 15);  // Show top 15 words

    // Find max absolute value for normalization
    const maxAbs = Math.max(...top.map(w => Math.abs(w.score)), 0.001);

    top.forEach(({ word, score }) => {
      const pct       = Math.min((Math.abs(score) / maxAbs) * 100, 100);
      const isPos     = score > 0;
      const scoreText = (score > 0 ? '+' : '') + score.toFixed(3);

      const row = document.createElement('div');
      row.className = 'word-attribution';
      row.innerHTML = `
        <span class="word-token" title="${escapeHtml(word)}">${escapeHtml(word)}</span>
        <div class="word-bar-container">
          <div
            class="word-bar ${isPos ? 'bar-positive' : 'bar-negative'}"
            style="width: ${pct}%"
          ></div>
        </div>
        <span class="word-score" style="color: ${isPos ? '#e53e3e' : '#3182ce'}">${scoreText}</span>
      `;
      explanationWords.appendChild(row);
    });
  }

  explanationPanel.classList.remove('hidden');
}


// ================================================================
// UI HELPERS
// ================================================================

function showLoading(show) {
  if (show) {
    loadingState.classList.remove('hidden');
    emptyState.classList.add('hidden');
    resultsList.classList.add('hidden');
  } else {
    loadingState.classList.add('hidden');
  }
}

function showEmptyState(message = null) {
  showLoading(false);
  emptyState.classList.remove('hidden');
  resultsList.classList.add('hidden');

  if (message) {
    const msgEl = emptyState.querySelector('p');
    if (msgEl) msgEl.textContent = message;
  }
}

function truncate(text, maxLen) {
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen - 3) + '...';
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.appendChild(document.createTextNode(text || ''));
  return div.innerHTML;
}


// ================================================================
// EVENT BINDING
// ================================================================

function bindEvents() {
  // Toggle enable/disable
  enabledToggle.addEventListener('change', async () => {
    const enabled = enabledToggle.checked;

    try {
      await chrome.runtime.sendMessage({
        type: 'TOGGLE_EXTENSION',
        enabled,
      });
    } catch (e) {
      console.error('[Popup] Toggle failed:', e);
    }
  });

  // Rescan button: ask content.js to re-analyze the page
  rescanBtn.addEventListener('click', async () => {
    rescanBtn.disabled = true;
    rescanBtn.textContent = '…';

    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (tab?.id) {
        // Inject a script to trigger re-analysis
        await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          func: () => {
            // This runs in the tab context
            // Re-trigger the analysis function if content.js is loaded
            if (typeof analyzeVisibleHeadlines === 'function') {
              analyzeVisibleHeadlines();
            }
          },
        });

        // Wait a moment for analysis to complete, then reload popup results
        setTimeout(async () => {
          await loadPageResults();
          await loadStoredStats();
          rescanBtn.disabled = false;
          rescanBtn.textContent = '↻ Rescan';
        }, 2000);
      }
    } catch (e) {
      console.error('[Popup] Rescan failed:', e);
      rescanBtn.disabled = false;
      rescanBtn.textContent = '↻ Rescan';
    }
  });

  // Close explanation panel
  closeExplanation.addEventListener('click', () => {
    explanationPanel.classList.add('hidden');
  });
}
