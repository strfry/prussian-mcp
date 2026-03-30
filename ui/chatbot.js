/**
 * Prussian Dictionary Chatbot - UI Layer with MCP Architecture
 * Handles DOM manipulation, events, display, and MCP orchestration
 */

let busy = false;
let currentLang = "de";
let debugMode = false;
let conversationHistory = [];
let mcpClient = null;
let chatEngine = null;
let currentStreamingContent = '';

const i18n = {
  de: {
    "intro.line1": "Sprich mich an — auf Deutsch, Litauisch oder Englisch.",
    "intro.line2": "Ich antworte auf Altpreußisch, mit deutscher Übersetzung.",
    "input.placeholder": "Schreib etwas auf Deutsch oder Englisch…",
    "typing": "Suche im Wörterbuch…",
    "error.prefix": "⚠ ",
    "hits.label": "📖 {count} Wörterbucheinträge",
    "translation.prefix": "🇩🇪",
    "connecting": "Verbinde mit Server…",
    "connected": "Verbunden",
    "disconnected": "Getrennt"
  },
  lt: {
    "intro.line1": "Pakalbėk su manimi – vokiečių, lietuvių arba anglų kalba.",
    "intro.line2": "Aš atsakau senovės prūsų kalba su lietuvių vertimu.",
    "input.placeholder": "Parašyk ką nors vokiečių arba anglų kalba…",
    "typing": "Ieško žodyne…",
    "error.prefix": "⚠ ",
    "hits.label": "📖 {count} žodyno įrašai",
    "translation.prefix": "🇱🇹",
    "connecting": "Jungiamasi prie serverio…",
    "connected": "Prisijungta",
    "disconnected": "Atsijungta"
  }
};

function t(key, vars = {}) {
  let text = i18n[currentLang]?.[key] || i18n["de"]?.[key] || key;
  for (const [k, v] of Object.entries(vars)) {
    text = text.replace(`{${k}}`, v);
  }
  return text;
}

function getLangIcon(lang) {
  return lang === "lt" ? "🇱🇹" : "🇩🇪";
}

function detectBrowserLanguage() {
  const stored = localStorage.getItem("preusDictLang");
  if (stored) return stored;

  const acceptLang = navigator.language || navigator.userLanguage || "";
  const langPref = acceptLang.toLowerCase();

  if (langPref.startsWith("lt")) return "lt";
  if (langPref.startsWith("de") || langPref.startsWith("at") || langPref.startsWith("ch")) return "de";
  if (langPref.length > 0 && !langPref.startsWith("en")) return "lt";

  return "de";
}

const chat = document.getElementById("chat");
const input = document.getElementById("input");
const btn = document.getElementById("send");
const errEl = document.getElementById("error");
const debugToggle = document.getElementById("debugToggle");
const langIcon = document.getElementById("lang-icon");
const langMenu = document.getElementById("lang-menu");
const langButtons = langMenu.querySelectorAll("button");
const statusEl = document.getElementById("status");

function clearEmpty() {
  const e = chat.querySelector(".empty");
  if (e) e.remove();
}

function showIntro() {
  clearEmpty();
  const intro = document.createElement("div");
  intro.className = "empty";
  intro.innerHTML = `<div class="icon">⚡</div>
    ${t("intro.line1")}<br>
    <span style="font-size:13px">${t("intro.line2")}</span>`;
  chat.appendChild(intro);
}

function addMsg(role, prussian, translation, words, debugInfo) {
  console.log(`addMsg called: role=${role}, words=${JSON.stringify(words)}`);
  clearEmpty();
  const div = document.createElement("div");
  div.className = "msg " + (role === "user" ? "user" : "bot");

  if (role === "bot") {
    div.innerHTML = `<div class="avatar">🏺</div>
      <div class="bubble">
        <div>${prussian.replace(/\n/g, "<br>")}</div>
        ${translation ? `<div class="translation">${t("translation.prefix")} ${translation}</div>` : ""}
      </div>`;

    if (Array.isArray(words) && words.length > 0) {
      const bubble = div.querySelector('.bubble');
      const hitsDiv = document.createElement('div');
      hitsDiv.className = 'dict-hits';
      hitsDiv.textContent = t("hits.label", { count: words.length });

      const tooltip = document.createElement('span');
      tooltip.className = 'tooltip';
      tooltip.textContent = words.join(', ');
      hitsDiv.appendChild(tooltip);
      bubble.appendChild(hitsDiv);
    }

    if (debugMode && debugInfo) {
      const bubble = div.querySelector('.bubble');
      const debugPanel = createDebugPanel(debugInfo);
      bubble.appendChild(debugPanel);
    }
  } else {
    div.innerHTML = `<div class="bubble">${prussian}</div>`;
  }

  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function addStreamingMsg() {
  clearEmpty();
  const div = document.createElement("div");
  div.className = "msg bot";
  div.id = "streaming-msg";
  div.innerHTML = `<div class="avatar">🏺</div>
    <div class="bubble">
      <div id="streaming-content"></div>
    </div>`;
  chat.appendChild(div);
  return div;
}

function updateStreamingMsg(delta) {
  const contentEl = document.getElementById("streaming-content");
  if (contentEl) {
    contentEl.innerHTML = (currentStreamingContent + delta).replace(/\n/g, "<br>");
    chat.scrollTop = chat.scrollHeight;
  }
}

function finalizeStreamingMsg(prussian, translation, words, debugInfo) {
  const streamingMsg = document.getElementById("streaming-msg");
  if (streamingMsg) {
    streamingMsg.innerHTML = `<div class="avatar">🏺</div>
      <div class="bubble">
        <div>${prussian.replace(/\n/g, "<br>")}</div>
        ${translation ? `<div class="translation">${t("translation.prefix")} ${translation}</div>` : ""}
      </div>`;

    if (Array.isArray(words) && words.length > 0) {
      const bubble = streamingMsg.querySelector('.bubble');
      const hitsDiv = document.createElement('div');
      hitsDiv.className = 'dict-hits';
      hitsDiv.textContent = t("hits.label", { count: words.length });

      const tooltip = document.createElement('span');
      tooltip.className = 'tooltip';
      tooltip.textContent = words.join(', ');
      hitsDiv.appendChild(tooltip);
      bubble.appendChild(hitsDiv);
    }

    if (debugMode && debugInfo) {
      const bubble = streamingMsg.querySelector('.bubble');
      const debugPanel = createDebugPanel(debugInfo);
      bubble.appendChild(debugPanel);
    }
  }
  currentStreamingContent = '';
}

function createDebugPanel(debugInfo) {
  const panel = document.createElement('div');
  panel.className = 'debug-panel';

  let html = '<details><summary>🔍 RAG Debug Info</summary>';

  html += `<div class="debug-section">
    <h4>User Query</h4>
    <div class="debug-query">"${debugInfo.query}"</div>
  </div>`;

  if (debugInfo.reasoning && debugInfo.reasoning.length > 0) {
    html += `<div class="debug-section">
      <h4>🧠 DeepSeek R1 Reasoning (${debugInfo.reasoning.length} turn${debugInfo.reasoning.length > 1 ? 's' : ''})</h4>`;

    debugInfo.reasoning.forEach((r, i) => {
      const reasoningText = r.reasoning || '';
      const preview = reasoningText.length > 200
        ? reasoningText.substring(0, 200) + '...'
        : reasoningText;

      html += `<details class="reasoning-turn">
        <summary><strong>Turn ${r.turn}</strong> (${reasoningText.length} chars) - ${preview}</summary>
        <div class="reasoning-content">${reasoningText.replace(/\n/g, '<br>')}</div>
      </details>`;
    });

    html += '</div>';
  }

  if (debugInfo.toolCalls && debugInfo.toolCalls.length > 0) {
    html += `<div class="debug-section">
      <h4>Tool Calls (${debugInfo.toolCalls.length})</h4>`;
    debugInfo.toolCalls.forEach((call, i) => {
      html += `<div class="debug-result">
        <strong>${i + 1}. ${call.name}</strong><br>
        <div style="margin-left:16px; margin-top:4px;">
          Input: <code>${JSON.stringify(call.input)}</code><br>
          Results: ${Array.isArray(call.result) ? call.result.length + ' entries' : 'object'}
        </div>
      </div>`;
    });
    html += '</div>';
  }

  if (debugInfo.results && debugInfo.results.length > 0) {
    html += `<div class="debug-section">
      <h4>All Dictionary Results (${debugInfo.results.length} total)</h4>`;
    debugInfo.results.slice(0, 10).forEach((r, i) => {
      const translations = r.de || r.en || ['?'];
      html += `<div class="debug-result">
        <strong>${r.word || '?'}</strong> — ${Array.isArray(translations) ? translations[0] : translations}
      </div>`;
    });
    if (debugInfo.results.length > 10) {
      html += `<div style="margin-top:8px; color:#999; font-style:italic;">
        ...and ${debugInfo.results.length - 10} more results
      </div>`;
    }
    html += '</div>';
  }

  if (debugInfo.usedWords && debugInfo.usedWords.length > 0) {
    html += `<div class="debug-section">
      <h4>Words Actually Used (${debugInfo.usedWords.length})</h4>
      <div>${debugInfo.usedWords.join(', ')}</div>
    </div>`;
  }

  html += '</details>';
  panel.innerHTML = html;
  return panel;
}

function showTyping() {
  const d = document.createElement("div");
  d.className = "typing";
  d.id = "typing";
  d.innerHTML = `<span>🏺</span><span>${t("typing")}</span>
    <div class="dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>`;
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
}

function hideTyping() {
  document.getElementById("typing")?.remove();
}

function showError(msg) {
  errEl.textContent = t("error.prefix") + msg;
  errEl.style.display = "block";
}

function hideError() {
  errEl.style.display = "none";
}

function setUI(enabled) {
  input.disabled = !enabled;
  btn.disabled = !enabled || !input.value.trim();
}

function updateUI() {
  langIcon.textContent = getLangIcon(currentLang);
  langButtons.forEach(b => {
    b.classList.toggle("active", b.dataset.lang === currentLang);
  });
  input.placeholder = t("input.placeholder");

  if (!chat.querySelector(".msg")) {
    showIntro();
  }
}

function updateStatus(status) {
  if (statusEl) {
    statusEl.textContent = status;
    statusEl.className = status.includes("Verbinde") ? "status connecting" :
                         status.includes("Verbunden") ? "status connected" : "status";
  }
}

async function send() {
  const text = input.value.trim();
  if (!text || busy) return;
  if (!mcpClient || !chatEngine) {
    showError("MCP not connected");
    return;
  }

  busy = true;
  hideError();
  input.value = "";
  setUI(false);

  addMsg("user", text);
  hideTyping();

  const streamingMsg = addStreamingMsg();

  try {
    const result = await chatEngine.sendMessage(text, currentLang, conversationHistory, {
      onContentDelta: (delta, full) => {
        currentStreamingContent = full;
        updateStreamingMsg(delta);
      },
      onToolCall: (toolInfo) => {
        console.log(`[UI] Tool call: ${toolInfo.name}`);
      },
      onToolResult: (result) => {
        console.log(`[UI] Tool result:`, result);
      },
      onDone: () => {
        hideTyping();
      }
    });

    finalizeStreamingMsg(result.prussian, result.translation, result.usedWords, result.debugInfo);
    addMsg = (() => {})();
    conversationHistory = [
      ...conversationHistory,
      { role: "user", content: text },
      { role: "assistant", content: result.prussian + (result.translation ? `\n\n[${currentLang.toUpperCase()}: ${result.translation}]` : '') }
    ];

  } catch (e) {
    if (document.getElementById("streaming-msg")) {
      document.getElementById("streaming-msg").remove();
    }
    showError(e.message);
  }

  busy = false;
  setUI(true);
  input.focus();
}

async function initMCP() {
  updateStatus(t("connecting"));

  try {
    mcpClient = new MCPClient('');
    await mcpClient.connect();

    chatEngine = new ChatEngine(mcpClient);

    updateStatus(t("connected"));
    setUI(true);
    input.focus();

    console.log('[UI] MCP initialized successfully');
  } catch (err) {
    console.error('[UI] MCP init failed:', err);
    updateStatus(t("disconnected"));
    showError("Failed to connect to MCP server: " + err.message);
  }
}

debugToggle.addEventListener("click", () => {
  debugMode = !debugMode;
  debugToggle.classList.toggle("active", debugMode);
  debugToggle.textContent = debugMode ? "🔍 Debug ON" : "🔍 Debug";
});

langIcon.addEventListener("click", () => {
  langMenu.classList.toggle("open");
});

langButtons.forEach(btn => {
  btn.addEventListener("click", () => {
    const newLang = btn.dataset.lang;
    if (newLang !== currentLang) {
      currentLang = newLang;
      localStorage.setItem("preusDictLang", newLang);
      updateUI();
    }
    langMenu.classList.remove("open");
  });
});

btn.addEventListener("click", send);
input.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) send();
});
input.addEventListener("input", () => {
  btn.disabled = !input.value.trim() || busy;
});

currentLang = detectBrowserLanguage();
updateUI();
showTyping();
initMCP();
