/**
 * background.js
 * =============
 * Chrome Extension Service Worker (Manifest V3).
 *
 * Responsibilities:
 *   1. Maintain extension enabled/disabled state in chrome.storage
 *   2. Listen for messages from content.js (headline analysis requests)
 *   3. Dynamically inject inference.js into tabs when needed
 *   4. Coordinate between content script and inference module
 *
 * PRIVACY NOTE:
 *   - This service worker never makes external network requests.
 *   - Headline text is processed locally and discarded after analysis.
 *   - No browsing history, no analytics, no telemetry.
 *   - All model inference happens inside the tab via onnxruntime-web.
 *
 * Architecture:
 *   content.js  →  (chrome.runtime.sendMessage)  →  background.js
 *   background.js →  (chrome.tabs.sendMessage)   →  content.js (with results)
 *
 *   Actual ML inference runs in inference.js which is injected into the page
 *   context. This gives it access to the ONNX model files as extension resources.
 */

'use strict';

// ================================================================
// INITIALIZATION
// Set default extension state when first installed
// ================================================================
chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === 'install') {
    chrome.storage.local.set({
      enabled: true,         // Extension active by default
      totalAnalyzed: 0,      // Count of headlines analyzed this session
      clickbaitFound: 0,     // Count of clickbait detected
    });
    console.log('[ClickbaitDetector] Extension installed and ready.');
  }
});


// ================================================================
// MESSAGE HANDLER
// Receives messages from content.js and routes them
// ================================================================
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {

  // Handle analysis requests from content.js
  if (message.type === 'ANALYZE_HEADLINES') {
    handleAnalysisRequest(message, sender, sendResponse);
    return true;  // Keep message channel open for async response
  }

  // Handle state toggle from popup.js
  if (message.type === 'TOGGLE_EXTENSION') {
    toggleExtension(message.enabled, sendResponse);
    return true;
  }

  // Handle stats request from popup.js
  if (message.type === 'GET_STATS') {
    chrome.storage.local.get(['totalAnalyzed', 'clickbaitFound', 'enabled'], (data) => {
      sendResponse({ success: true, data });
    });
    return true;
  }

  // Handle request to update stats from content.js
  if (message.type === 'UPDATE_STATS') {
    updateStats(message.analyzed, message.clickbait);
    return true;
  }
});


/**
 * handleAnalysisRequest
 * ---------------------
 * Routes headline analysis request.
 * Tells content.js to proceed with local inference.
 *
 * In MV3, we can't run onnxruntime-web directly in the service worker
 * (no DOM access). Instead, inference runs in the content script context
 * via the injected inference.js module.
 *
 * @param {object} message - { type, headlines, tabId }
 * @param {object} sender  - Chrome sender info (includes tab info)
 * @param {function} sendResponse - Callback to send response to content.js
 */
async function handleAnalysisRequest(message, sender, sendResponse) {
  // Check if extension is enabled
  const { enabled } = await chrome.storage.local.get('enabled');
  if (!enabled) {
    sendResponse({ success: false, reason: 'Extension is disabled.' });
    return;
  }

  // Validate input
  if (!message.headlines || message.headlines.length === 0) {
    sendResponse({ success: false, reason: 'No headlines provided.' });
    return;
  }

  // Tell content.js to run inference locally
  // The actual ONNX inference runs inside inference.js in the tab context
  sendResponse({
    success: true,
    command: 'RUN_INFERENCE',
    headlines: message.headlines,
  });
}


/**
 * toggleExtension
 * ---------------
 * Enable or disable the clickbait detection globally.
 * State persists across browser sessions via chrome.storage.
 */
async function toggleExtension(enabled, sendResponse) {
  await chrome.storage.local.set({ enabled });
  console.log(`[ClickbaitDetector] Extension ${enabled ? 'enabled' : 'disabled'}`);

  // Notify all active tabs of state change
  const tabs = await chrome.tabs.query({ active: true });
  for (const tab of tabs) {
    try {
      chrome.tabs.sendMessage(tab.id, {
        type: 'EXTENSION_STATE_CHANGED',
        enabled,
      });
    } catch (e) {
      // Tab may not have content script — ignore
    }
  }

  sendResponse({ success: true, enabled });
}


/**
 * updateStats
 * -----------
 * Increment running counters stored in chrome.storage.
 * Used by content.js after batch analysis completes.
 */
async function updateStats(analyzed, clickbait) {
  const { totalAnalyzed, clickbaitFound } = await chrome.storage.local.get([
    'totalAnalyzed',
    'clickbaitFound',
  ]);
  await chrome.storage.local.set({
    totalAnalyzed:  (totalAnalyzed  || 0) + (analyzed  || 0),
    clickbaitFound: (clickbaitFound || 0) + (clickbait || 0),
  });
}


// ================================================================
// ERROR HANDLING
// Log unhandled errors in the service worker
// ================================================================
self.addEventListener('unhandledrejection', (event) => {
  console.error('[ClickbaitDetector] Unhandled promise rejection:', event.reason);
});
