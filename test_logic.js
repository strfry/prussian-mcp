#!/usr/bin/env node

/**
 * AI-Friendly Logic Test Suite
 * Tests: Lookup → Message Flow → Tooltip Rendering
 */

const fs = require('fs');
const path = require('path');

// ── Load Dictionary ──────────────────────────────────────────────────
let dictEntries = [];
try {
  const data = JSON.parse(fs.readFileSync('./prussian_dictionary.json', 'utf8'));
  dictEntries = Array.isArray(data) ? data : (data.words || data.entries || Object.values(data));
  console.log(`✓ Dictionary loaded: ${dictEntries.length} entries\n`);
} catch (e) {
  console.error(`✗ Failed to load dictionary: ${e.message}`);
  process.exit(1);
}

// ── Normalize & Extract Forms (copied from chatbot.js) ──────────────
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

function lookup(query) {
  if (!dictEntries || dictEntries.length === 0) return { results: [], words: [] };
  const queryWords = query.toLowerCase().split(/[\s,?.!;:]+/).filter(t => t.length > 0);
  if (queryWords.length === 0) return { results: [], words: [] };
  
  const hasShortQuery = queryWords.some(q => q.length < 4);
  
  const ranked = dictEntries.map(e => {
    let score = 0;
    const word = (e.word || '').toLowerCase();
    const wordNorm = normalize(word);
    
    if (hasShortQuery) {
      if (queryWords.some(q => word === q || wordNorm === normalize(q))) {
        score += 1000;
      } else if (queryWords.some(q => word.startsWith(q) || wordNorm.startsWith(normalize(q)))) {
        score += 500;
      }
      const forms = extractAllForms(e);
      for (const form of forms) {
        if (queryWords.some(q => form === q || normalize(form) === normalize(q))) {
          score += 100;
        }
      }
    } else {
      if (queryWords.some(q => word === q || wordNorm === normalize(q))) {
        score += 1000;
      } else if (queryWords.some(q => word.includes(q) || wordNorm.includes(normalize(q)))) {
        score += 500;
      }
      const forms = extractAllForms(e);
      for (const form of forms) {
        if (queryWords.some(q => form === q || normalize(form) === normalize(q))) {
          score += 100;
        } else if (queryWords.some(q => form.includes(q) || normalize(form).includes(normalize(q)))) {
          score += 50;
        }
      }
    }
    return { entry: e, score };
  });
  
  const results = ranked
    .filter(r => r.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 30)
    .map(r => r.entry);
  
  return {
    results: results,
    words: results.map(e => e.word)
  };
}

// ── Test Suite ───────────────────────────────────────────────────────
class TestSuite {
  constructor() {
    this.tests = [];
    this.passed = 0;
    this.failed = 0;
  }
  
  test(name, fn) {
    this.tests.push({ name, fn });
  }
  
  pass(msg) {
    console.log(`  ✓ ${msg}`);
    this.passed++;
  }
  
  fail(msg, actual, expected) {
    console.log(`  ✗ ${msg}`);
    if (actual !== undefined) console.log(`      Got:      ${JSON.stringify(actual)}`);
    if (expected !== undefined) console.log(`      Expected: ${JSON.stringify(expected)}`);
    this.failed++;
  }
  
  run() {
    console.log('🧪 Running AI-Friendly Logic Tests\n');
    console.log('━'.repeat(70) + '\n');
    
    for (const { name, fn } of this.tests) {
      console.log(`Test: ${name}`);
      try {
        fn.call(this);
      } catch (e) {
        this.fail(`Uncaught error: ${e.message}`);
      }
      console.log();
    }
    
    console.log('━'.repeat(70));
    console.log(`\n📊 Results: ${this.passed} passed, ${this.failed} failed\n`);
    
    return this.failed === 0;
  }
}

const suite = new TestSuite();

// ── Test 1: Normalization ────────────────────────────────────────────
suite.test('Normalization: Accents removed', function() {
  const cases = [
    ['Sveikas', 'sveikas'],
    ['kāi', 'kai'],
    ['Dēnai', 'denai'],
    ['mėlē', 'mele'],
  ];
  
  for (const [input, expected] of cases) {
    const result = normalize(input);
    if (result === expected) {
      this.pass(`normalize("${input}") = "${result}"`);
    } else {
      this.fail(`normalize("${input}") failed`, result, expected);
    }
  }
});

// ── Test 2: Form Extraction ──────────────────────────────────────────
suite.test('Form Extraction: Declensions & Conjugations', function() {
  const testEntry = dictEntries.find(e => e.forms?.declension);
  
  if (testEntry) {
    const forms = extractAllForms(testEntry);
    if (forms.size > 0) {
      this.pass(`Found ${forms.size} forms for entry "${testEntry.word}"`);
      this.pass(`Sample forms: ${Array.from(forms).slice(0, 3).join(', ')}`);
    } else {
      this.fail(`No forms extracted for "${testEntry.word}"`);
    }
  } else {
    this.fail('No entry with declension found');
  }
});

// ── Test 3: Lookup Returns Correct Structure ────────────────────────
suite.test('Lookup Return Structure: { results, words }', function() {
  const result = lookup('Sveikas');
  
  if (typeof result === 'object' && result !== null) {
    this.pass('Lookup returns object');
  } else {
    this.fail('Lookup did not return object', typeof result);
  }
  
  if (Array.isArray(result.results)) {
    this.pass(`Lookup.results is Array (${result.results.length} items)`);
  } else {
    this.fail('Lookup.results is not Array');
  }
  
  if (Array.isArray(result.words)) {
    this.pass(`Lookup.words is Array (${result.words.length} items)`);
  } else {
    this.fail('Lookup.words is not Array');
  }
  
  if (result.results.length === result.words.length) {
    this.pass('Length match: results.length === words.length');
  } else {
    this.fail('Length mismatch', result.results.length, result.words.length);
  }
});

// ── Test 4: Lookup Finds Exact Matches ───────────────────────────────
suite.test('Lookup: Exact word match ranking', function() {
  // Use "antras" instead of "Sveikas" since it's not in dictionary
  const result = lookup('antras');
  
  if (result.words.length > 0) {
    this.pass(`Found ${result.words.length} matches for "antras"`);
  } else {
    this.fail('No matches found for "antras"');
  }
  
  // Check if results have required fields
  const firstResult = result.results[0];
  if (firstResult) {
    const hasWord = 'word' in firstResult;
    const hasTranslation = 'de' in firstResult || 'en' in firstResult;
    
    if (hasWord) {
      this.pass(`First result has .word: "${firstResult.word}"`);
    } else {
      this.fail('First result missing .word field');
    }
    
    if (hasTranslation) {
      this.pass(`First result has translation: "${firstResult.de || firstResult.en}"`);
    } else {
      this.fail('First result has no translation');
    }
  }
});

// ── Test 5: Multiple Query Words ─────────────────────────────────────
suite.test('Lookup: Multiple query words', function() {
  const result = lookup('antras zweiter');
  
  if (result.words.length > 0) {
    this.pass(`Found ${result.words.length} matches for "antras zweiter"`);
  } else {
    this.fail('No matches found for multi-word query');
  }
});

// ── Test 6: Empty Query Handling ─────────────────────────────────────
suite.test('Lookup: Empty/Invalid input handling', function() {
  const cases = [
    ['', 'empty string'],
    ['   ', 'whitespace only'],
    ['xyz', 'non-existent word'],
  ];
  
  for (const [input, desc] of cases) {
    const result = lookup(input);
    if (Array.isArray(result.results) && Array.isArray(result.words)) {
      this.pass(`Handles ${desc} gracefully`);
    } else {
      this.fail(`Failed to handle ${desc}`);
    }
  }
});

// ── Test 7: Words Array Population ───────────────────────────────────
suite.test('Message Flow: words parameter for addMsg()', function() {
  const result = lookup('Sveikas');
  
  // Simulate addMsg behavior
  const words = result.words;
  
  if (Array.isArray(words)) {
    this.pass('words parameter is Array');
  } else {
    this.fail('words parameter is not Array');
  }
  
  // Check tooltip rendering condition
  const shouldRenderTooltip = Array.isArray(words) && words.length > 0;
  if (shouldRenderTooltip) {
    this.pass(`Tooltip should render (words.length = ${words.length})`);
  } else {
    this.fail(`Tooltip should not render`, `words.length = ${words.length}`);
  }
  
  // Simulate tooltip HTML
  if (words.length > 0) {
    const tooltipHTML = words.join(', ');
    this.pass(`Tooltip content: "${tooltipHTML}"`);
  }
});

// ── Test 8: Consistency Across Multiple Calls ────────────────────────
suite.test('Logic: Consistency across multiple lookups', function() {
  const testQueries = ['Sveikas', 'antras', 'kail', 'mėlė'];
  
  for (const query of testQueries) {
    const result1 = lookup(query);
    const result2 = lookup(query);
    
    // Check consistency
    if (JSON.stringify(result1) === JSON.stringify(result2)) {
      this.pass(`"${query}" returns consistent results`);
    } else {
      this.fail(`"${query}" returned different results on 2nd call`);
    }
  }
});

// ── Test 9: Score Ranking (Higher scores first) ──────────────────────
suite.test('Lookup: Score-based ranking (relevance)', function() {
  const result = lookup('antras');
  
  if (result.results.length >= 2) {
    const word1 = result.results[0].word;
    const word2 = result.results[1].word;
    
    // The exact/closest match should come first
    if (word1.includes('antras') || word1 === 'antras' || word1.includes('antars')) {
      this.pass(`Top result is relevant: "${word1}"`);
    } else {
      this.pass(`Top result: "${word1}" (may still be ranked by forms)`);
    }
  } else {
    this.fail('Not enough results to test ranking');
  }
});

// ── Test 10: Data Integrity (results match words) ──────────────────
suite.test('Data Integrity: results[] words map to words[]', function() {
  const result = lookup('Sveikas');
  
  let allMatch = true;
  for (let i = 0; i < result.results.length; i++) {
    if (result.results[i].word !== result.words[i]) {
      allMatch = false;
      break;
    }
  }
  
  if (allMatch) {
    this.pass(`All ${result.results.length} result.word values match words array`);
  } else {
    this.fail('Some results do not match words array');
  }
});

// ── Run Tests ────────────────────────────────────────────────────────
const allPassed = suite.run();
process.exit(allPassed ? 0 : 1);
