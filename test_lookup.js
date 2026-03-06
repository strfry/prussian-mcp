#!/usr/bin/env node
/**
 * Test script for dictionary lookup and ranking
 * Usage: node test_lookup.js <query> [max_results]
 */

const fs = require('fs');
const path = require('path');

// Load dictionary
const dictPath = path.join(__dirname, 'prussian_dictionary.json');
const dictEntries = JSON.parse(fs.readFileSync(dictPath, 'utf-8'));

// ── Copy functions from chatbot.html ──────────────────────────────────
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

function lookup(query, maxHits = 30) {
  if (!dictEntries || dictEntries.length === 0) return { results: [], words: [], debug: { totalMatches: 0, ranked: [] } };
  
  const queryWords = query.toLowerCase().split(/[\s,?.!;:]+/).filter(t => t.length > 0);
  if (queryWords.length === 0) return { results: [], words: [], debug: { totalMatches: 0, ranked: [] } };
  
  // Check if any query word is short (< 4 chars) → use prefix matching
  const hasShortQuery = queryWords.some(q => q.length < 4);
  
  // Rank results by relevance (exact > forms > translations)
  const ranked = dictEntries.map(e => {
    let score = 0;
    const word = (e.word || '').toLowerCase();
    const wordNorm = normalize(word);
    
    let matchType = null;
    
    // For short queries: strict prefix matching on lemma
    if (hasShortQuery) {
      if (queryWords.some(q => word === q || wordNorm === normalize(q))) {
        score += 1000;
        matchType = 'exact_lemma';
      }
      else if (queryWords.some(q => word.startsWith(q) || wordNorm.startsWith(normalize(q)))) {
        score += 500;
        matchType = 'lemma_prefix';
      }
      
      // Still match in forms/translations
      const forms = extractAllForms(e);
      for (const form of forms) {
        if (queryWords.some(q => form === q || normalize(form) === normalize(q))) {
          score += 100;
          if (!matchType) matchType = 'exact_form';
        }
      }
      
      for (const lang of ['miks', 'engl', 'leit']) {
        const translations = e.translations?.[lang] || [];
        for (const trans of translations) {
          const tLower = trans.toLowerCase();
          const words = tLower.split(/[\s,;]+/);
          for (const w of words) {
            if (queryWords.some(q => w === q)) {
              score += 20;
              if (!matchType) matchType = 'exact_translation';
            }
          }
        }
      }
    } else {
      // For longer queries: normal substring matching
      if (queryWords.some(q => word === q || wordNorm === normalize(q))) {
        score += 1000;
        matchType = 'exact_lemma';
      }
      else if (queryWords.some(q => word.includes(q) || wordNorm.includes(normalize(q)))) {
        score += 500;
        matchType = 'lemma_substring';
      }
      
      const forms = extractAllForms(e);
      for (const form of forms) {
        if (queryWords.some(q => form === q || normalize(form) === normalize(q))) {
          score += 100;
          if (!matchType) matchType = 'exact_form';
        }
        else if (queryWords.some(q => form.includes(q) || normalize(form).includes(normalize(q)))) {
          score += 50;
          if (!matchType) matchType = 'form_substring';
        }
      }
      
      for (const lang of ['miks', 'engl', 'leit']) {
        const translations = e.translations?.[lang] || [];
        for (const trans of translations) {
          const tLower = trans.toLowerCase();
          const words = tLower.split(/[\s,;]+/);
          for (const w of words) {
            if (queryWords.some(q => w === q)) {
              score += 20;
              if (!matchType) matchType = 'exact_translation';
            }
            else if (queryWords.some(q => w.includes(q))) {
              score += 5;
              if (!matchType) matchType = 'translation_substring';
            }
          }
        }
      }
    }
    
    return { entry: e, score, matchType };
  });
  
  // Filter and sort by score
  const filtered = ranked.filter(r => r.score > 0);
  const sorted = filtered.sort((a, b) => b.score - a.score);
  const final = sorted.slice(0, maxHits).map(r => r.entry);
  
  return {
    results: final,
    words: final.map(e => e.word),
    debug: {
      query,
      queryWords,
      totalMatches: filtered.length,
      topMatches: sorted.slice(0, 10).map(r => ({
        word: r.entry.word,
        score: r.score,
        type: r.matchType,
        de: r.entry.translations?.miks?.[0],
        en: r.entry.translations?.engl?.[0]
      }))
    }
  };
}

// ── Test ──────────────────────────────────────────────────────────────
const query = process.argv[2] || 'kails';
const maxHits = parseInt(process.argv[3], 10) || 30;

console.log(`\n🔍 Testing lookup: "${query}"\n`);

const result = lookup(query, maxHits);

console.log(`Results: ${result.words.length} hits (of ${result.debug.totalMatches} total matches)`);
console.log(`\nTop 10 matches (by relevance):`);
console.log('─'.repeat(80));

result.debug.topMatches.forEach((m, i) => {
  console.log(`${i+1}. "${m.word}" [${m.type}] score=${m.score}`);
  if (m.de) console.log(`   DE: ${m.de}`);
  if (m.en) console.log(`   EN: ${m.en}`);
  console.log('');
});

if (result.debug.totalMatches > 10) {
  console.log(`… and ${result.debug.totalMatches - 10} more matches`);
}
