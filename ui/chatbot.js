/**
 * Prussian Dictionary Chatbot - Main Application
 * All-in-one script with i18n, lookup, and UI logic
 */

// ── Configuration ────────────────────────────────────────────────────────
const DICT_URL = "/data/prussian_dictionary.json";
const LLM_URL = "/api/llm";            // LLM proxy endpoint
const API_SEARCH_URL = "/api/search";  // Semantic search endpoint (E5)
const API_FORMS_URL = "/api/forms";    // Forms lookup endpoint
const MAX_HITS = 30;

let dictEntries = [];
let history = [];
let busy = false;
let currentLang = "de";

// ── Normalization & Forms ────────────────────────────────────────────────
function normalize(text) {
  if (!text) return '';
  const replacements = {'ā':'a','ē':'e','ī':'i','ō':'o','ū':'u','ã':'a','ẽ':'e','ĩ':'i','õ':'o','ũ':'u'};
  return text.toLowerCase().split('').map(c => replacements[c] || c).join('');
}

function extractAllForms(entry) {
  const forms = new Set();
  
  if (entry.forms?.declension) {
    entry.forms.declension.forEach(decl => {
      (decl.cases || []).forEach(c => {
        if (c.singular) forms.add(c.singular.toLowerCase());
        if (c.plural) forms.add(c.plural.toLowerCase());
      });
    });
  }
  
  if (entry.forms?.indicative && Array.isArray(entry.forms.indicative)) {
    entry.forms.indicative.forEach(tenseGroup => {
      if (Array.isArray(tenseGroup.forms)) {
        tenseGroup.forms.forEach(f => {
          if (f && f.form) forms.add(f.form.toLowerCase());
        });
      }
    });
  }
  
  ['subjunctive', 'optative', 'imperative'].forEach(mood => {
    if (entry.forms?.[mood] && Array.isArray(entry.forms[mood])) {
      entry.forms[mood].forEach(item => {
        if (item && item.form) forms.add(item.form.toLowerCase());
      });
    }
  });
  
  ['participles', 'infinitives'].forEach(type => {
    if (entry.forms?.[type]) {
      entry.forms[type].forEach(item => {
        if (item.form) forms.add(item.form.toLowerCase());
      });
    }
  });
  
  return forms;
}

// ── Internationalization ─────────────────────────────────────────────────
const i18n = {
  de: {
    "dict.loading": "● Lade Wörterbuch…",
    "dict.loaded": "● Wörterbuch: {count} Einträge (mit allen Formen)",
    "dict.error": "● Wörterbuch: Fehler – {error}",
    "intro.line1": "Sprich mich an — auf Deutsch, Litauisch oder Englisch.",
    "intro.line2": "Ich antworte auf Altpreußisch, mit deutscher Übersetzung.",
    "input.placeholder": "Schreib etwas auf Deutsch oder Englisch…",
    "typing": "Suche im Wörterbuch…",
    "error.prefix": "⚠ ",
    "hits.label": "📖 {count} Wörterbucheinträge",
    "translation.prefix": "🇩🇪",
    "system.block": "## Wörterbuch-Einträge für diese Antwort",
    "system.intro": "Du bist ein Chatbot, der auf rekonstruiertem Neo-Preußisch (Altpreußisch) antwortet, gemäß der Rekonstruktion von Letas Palmaitis.",
    "system.rules": "Regeln:\n1. Nutze die Wörterbuch-Einträge unten für grammatisch korrekte Sätze. Verwende korrekte Kasus, Tempus und Numerus aus den bereitgestellten Formen.\n2. Fülle lexikalische Lücken mit baltischen Kognaten (litauisch/lettisch), angepasst an altpreußische Phonologie.\n3. Beende IMMER mit einer deutschen Übersetzung: [DE: …]\n4. Bei unsicheren Formen: kurze Anmerkung (rekonstruiert: …)\n5. 1–4 preußische Sätze. Warmer, leicht archaischer Ton.\n6. Keine langen linguistischen Erklärungen — einfach sprechen.",
    "system.short_words": "## Häufige kurze Wörter (immer verfügbar):\nak=ach|as=ich|be=ohne|di=es|din=sie|dis=er|gi=doch|ik=wenn|iz=aus|jā=ja|jāu=schon|jūs=ihr|kas=wer|me=wir|na=auf|ni=nicht|nū=nun|pa=unter|pas=nach|pat=eben|pra=durch|sēn=mit|sēr=Herz|tēr=nur|tēt=so|tū=du|uts=Laus|aks=Auge|ass=Achse|iws=Eibe|ēr=bis|ēn=in|īr=sogar|šis=dieser",
  },
  lt: {
    "dict.loading": "● Kraunasi žodynas…",
    "dict.loaded": "● Žodynas: {count} įrašų (su visomis formomis)",
    "dict.error": "● Žodynas: Klaida – {error}",
    "intro.line1": "Pakalbėk su manimi – vokiečių, lietuvių arba anglų kalba.",
    "intro.line2": "Aš atsakau senovės prūsų kalba su lietuvių vertimu.",
    "input.placeholder": "Parašyk ką nors vokiečių arba anglų kalba…",
    "typing": "Ieško žodyne…",
    "error.prefix": "⚠ ",
    "hits.label": "📖 {count} žodyno įrašai",
    "translation.prefix": "🇱🇹",
    "system.block": "## Šioms atsakymui naudoti žodyno įrašai",
    "system.intro": "Tu esi čatboto, kuris atsako senovės prūsų (Neo-Prussian) kalba pagal Letos Palmaičio rekonstrukciją.",
    "system.rules": "Taisyklės:\n1. Naudok žemiau pateiktus žodyno įrašus gramatiškai teisingiems sakingiams. Naudok teisingus kaip, laiką ir skaičių iš pateiktų formų.\n2. Užpildyk leksikines spragas baltų kognačiais (lietuvių/latvių), pritaikant senovės prūsų fonologijai.\n3. VISADA užbaigk lietuvių vertimu: [LT: …]\n4. Neaiškiems formoms: trumpa pastaba (rekonstruota: …)\n5. 1–4 prūsų sakiniai. Šiltas, šiek tiek archaiškas tono.\n6. Nėra ilgų lingvistinių paaiškini — tiesiog kalbėk.",
    "system.short_words": "## Dažniausi trumpi žodžiai (visuomet galimi):\nak=ak|as=aš|be=be|di=tai|din=ją|dis=jis|gi=juk|ik=jei|iz=iš|jā=taip|jāu=jau|jūs=jūs|kas=kas|me=mes|na=ant|ni=ne|nū=na|pa=po|pas=pãskui|pat=pat|pra=per|sēn=su|sēr=širdis|tēr=tik|tēt=taip|tū=tu|uts=utelė|aks=akis|ass=ašis|iws=pelėda|ēr=iki|ēn=į|īr=ir|šis=šis",
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

// ── Semantic Search API (E5) ─────────────────────────────────────────────
async function semanticSearch(query, topK = 30) {
  try {
    const response = await fetch(API_SEARCH_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k: topK })
    });

    if (!response.ok) {
      console.warn(`Semantic search API error: ${response.status}`);
      return null;
    }

    const data = await response.json();

    // Format API response to match old lookup() format
    const mapped = data.results.map(entry => {
      const cas = entry.forms?.declension?.[0]?.cases || [];
      const f = n => cas.find(c => c.case === n);
      return {
        word: entry.word,
        paradigm: entry.paradigm,
        gender: entry.gender || undefined,
        desc: entry.desc || undefined,
        de: entry.translations?.miks,
        en: entry.translations?.engl,
        lt: entry.translations?.leit,
        nom_sg: f("Nominative")?.singular,
        gen_sg: f("Genitive")?.singular,
        dat_sg: f("Dative")?.singular,
        acc_sg: f("Accusative")?.singular,
        nom_pl: f("Nominative")?.plural,
        gen_pl: f("Genitive")?.plural,
        dat_pl: f("Dative")?.plural,
        acc_pl: f("Accusative")?.plural,
        present: entry.forms?.indicative?.[0]?.forms,
        past: entry.forms?.indicative?.[1]?.forms,
        imperative: entry.forms?.imperative,
      };
    }).filter(e => e.de || e.en);

    return {
      results: mapped,
      words: mapped.map(r => r.word)
    };
  } catch (err) {
    console.error("Semantic search failed:", err);
    return null;
  }
}

async function getForms(lemma) {
  try {
    const response = await fetch(API_FORMS_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lemma })
    });

    if (!response.ok) {
      console.warn(`Forms lookup API error: ${response.status}`);
      return null;
    }

    return await response.json();
  } catch (err) {
    console.error("Forms lookup failed:", err);
    return null;
  }
}

// ── UI Elements & Helpers ────────────────────────────────────────────────
const chat = document.getElementById("chat");
const input = document.getElementById("input");
const btn = document.getElementById("send");
const errEl = document.getElementById("error");
const status = document.getElementById("dict-status");
const langIcon = document.getElementById("lang-icon");
const langMenu = document.getElementById("lang-menu");
const langButtons = langMenu.querySelectorAll("button");

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

function addMsg(role, prussian, german, words) {
  console.log(`addMsg called: role=${role}, words=${JSON.stringify(words)}`);
  clearEmpty();
  const div = document.createElement("div");
  div.className = "msg " + (role === "user" ? "user" : "bot");
  
  if (role === "bot") {
    div.innerHTML = `<div class="avatar">🏺</div>
      <div class="bubble">
        <div>${prussian.replace(/\n/g, "<br>")}</div>
        ${german ? `<div class="translation">${t("translation.prefix")} ${german}</div>` : ""}
      </div>`;
    
    console.log(`  → Checking: Array.isArray(words)=${Array.isArray(words)}, words.length=${words?.length}`);
    if (Array.isArray(words) && words.length > 0) {
      console.log(`  ✓ Creating tooltip for ${words.length} words`);
      const bubble = div.querySelector('.bubble');
      const hitsDiv = document.createElement('div');
      hitsDiv.className = 'dict-hits';
      hitsDiv.textContent = t("hits.label", { count: words.length });
      
      const tooltip = document.createElement('span');
      tooltip.className = 'tooltip';
      tooltip.textContent = words.join(', ');
      hitsDiv.appendChild(tooltip);
      
      bubble.appendChild(hitsDiv);
      console.log(`  ✓ Tooltip appended to bubble`);
    } else {
      console.log(`  ✗ Skipping tooltip: words is not array or empty`);
    }
  } else {
    div.innerHTML = `<div class="bubble">${prussian}</div>`;
  }
  
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
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

function parseMsg(text) {
  const deMatch = text.match(/\[DE:\s*([\s\S]*?)\]$/m);
  const ltMatch = text.match(/\[LT:\s*([\s\S]*?)\]$/m);
  const translation = deMatch?.[1] || ltMatch?.[1];
  const prussian = text.replace(/\[(DE|LT):\s*[\s\S]*?\]$/m, "").trim();
  
  return { prussian, german: translation?.trim() };
}

function setUI(enabled) {
  input.disabled = !enabled;
  btn.disabled = !enabled || !input.value.trim();
}

function updateUI() {
  langIcon.textContent = getLangIcon(currentLang);
  langButtons.forEach(btn => {
    btn.classList.toggle("active", btn.dataset.lang === currentLang);
  });
  input.placeholder = t("input.placeholder");
  
  if (!chat.querySelector(".msg")) {
    showIntro();
  }
  
  if (dictEntries.length > 0) {
    status.textContent = t("dict.loaded", { count: dictEntries.length.toLocaleString() });
  }
}

// ── API & Core Logic ─────────────────────────────────────────────────────
function makeSystem(hits) {
  const block = hits.length
    ? "\n\n" + t("system.block") + "\n" + JSON.stringify(hits, null, 1)
    : "";
  const shortWords = "\n\n" + t("system.short_words");
  return t("system.intro") + "\n\n" + t("system.rules") + shortWords + block;
}

async function callAPI(system) {
  const res = await fetch(LLM_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1000,
      system,
      messages: history.map(({ role, content }) => ({ role, content })),
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error?.message || `HTTP ${res.status}`);
  }
  const data = await res.json();
  return data.content?.filter(b => b.type === "text").map(b => b.text).join("\n") || "";
}

async function send() {
  const text = input.value.trim();
  if (!text || busy) return;
  busy = true;
  hideError();
  input.value = "";
  setUI(false);

  addMsg("user", text);
  history.push({ role: "user", content: text });

  // Use E5 semantic search (no fallback)
  const lookup_result = await semanticSearch(text, MAX_HITS);
  if (!lookup_result) {
    hideTyping();
    showError('Semantic search failed');
    busy = false;
    setUI(true);
    return;
  }
  console.log('✨ Semantic search result:', lookup_result);
  console.log('  → words:', lookup_result.words);
  showTyping();

  try {
    const raw = await callAPI(makeSystem(lookup_result.results));
    hideTyping();
    const { prussian, german } = parseMsg(raw);
    history.push({ role: "assistant", content: raw });
    console.log('📝 Calling addMsg with words:', lookup_result.words);
    addMsg("bot", prussian, german, lookup_result.words);
  } catch (e) {
    hideTyping();
    showError(e.message);
    history.pop();
  }

  busy = false;
  setUI(true);
  input.focus();
}

// ── Event Listeners ──────────────────────────────────────────────────────
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

// ── Initialization ───────────────────────────────────────────────────────
currentLang = detectBrowserLanguage();
updateUI();

status.textContent = t("dict.loading");
fetch(DICT_URL)
  .then(r => {
    if (!r.ok) throw new Error("HTTP " + r.status);
    return r.json();
  })
  .then(data => {
    dictEntries = Array.isArray(data) ? data : (data.words || data.entries || Object.values(data));
    status.style.color = "#5a9";
    status.textContent = t("dict.loaded", { count: dictEntries.length.toLocaleString() });
    setUI(true);
    input.focus();
  })
  .catch(e => {
    status.style.color = "#c55";
    status.textContent = t("dict.error", { error: e.message });
    setUI(true);
  });
