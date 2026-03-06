---
name: prussian-dictionary
description: Search the Old Prussian (Prußisch) dictionary with 4800+ entries. Search by Prussian word (lemma or any inflected form), or translate from English/German/Lithuanian/Latvian/Polish/Russian. Returns full entries with declensions, conjugations, translations, and grammatical metadata. Use for Prussian language work, translation, or Baltic linguistics.
---

# Prussian Dictionary (Prūsisks Wirdeīns)

Local dictionary of the Prussian language (Old Prussian / Prußisch), a Western Baltic language. Contains 4816 entries with full morphological data.

## Search for Words

```bash
cd /home/strfry/prussian-dictionary
./scripts/search.sh "<query>" [language] [max_results]
```

**Languages:** 
- `engl` (English, default)
- `miks` (German)
- `leit` (Lithuanian)
- `latt` (Latvian)
- `pols` (Polish)
- `mask` (Russian)
- `prus` (Prussian - searches in lemmas and ALL inflected forms)

**Max Results:** Number of entries to return (default: 20)

### Search Features

The search is **flexible** and finds matches in:
- Prussian lemmas (base words)
- **All inflected forms** (declensions, conjugations, participles, etc.)
- Translations in the specified language
- Supports fuzzy matching (diacritics optional: `a/ā`, `u/ū`, `i/ī`, `o/ō`)

### Examples

```bash
./scripts/search.sh "water" engl          # English → Prussian
./scripts/search.sh "Wasser" miks         # German → Prussian
./scripts/search.sh "wundan" prus         # Prussian lemma
./scripts/search.sh "wundans" prus        # Finds by inflected form (Acc.Pl)
./scripts/search.sh "deinaalgenīkamans"   # Finds by Dat.Pl form
```

### Output Format

Returns **complete JSON entries** with:
- `word`: Prussian lemma
- `paradigm`: Inflection class number (e.g., "32", "116")
- `gender`: masc, fem, neut (for nouns)
- `desc`: Source description (e.g., "[Deināalgenikamans 95 drv]")
- `translations`: Object with arrays for each language (engl, miks, leit, latt, pols, mask)
- `forms`: Complete declension/conjugation data
  - `declension`: Cases (Nominative, Genitive, Dative, Accusative) × Number (singular, plural)
  - `indicative`, `subjunctive`, `optative`, `imperative`: Verb moods with tense forms
  - `participles`, `infinitives`: Non-finite forms

## Get Forms for a Specific Word

If you need to retrieve the complete entry for a known word:

```bash
cd /home/strfry/prussian-dictionary
./scripts/forms.sh "<word>" [paradigm_number]
```

**Paradigm number** is optional and used for disambiguation when multiple entries exist.

### Examples

```bash
./scripts/forms.sh "wundan"               # Get full entry
./scripts/forms.sh "deinaalgenīks" 32     # Specific paradigm
```

Returns the full JSON entry with all forms and metadata.

## Usage Tips

1. **For translation**: Search with the source language parameter (engl, miks, etc.)
2. **For inflected forms**: Use `prus` language to search any Prussian form
3. **Flexible matching**: Diacritics are optional, substring matches work
4. **Rich context**: Every result includes full morphology - perfect for LLM context

## Data Structure

The dictionary (`prussian_dictionary.json`) contains 4816 entries. Each entry can include:
- Multiple translations per language
- Full paradigm (declension or conjugation tables)
- Gender, paradigm class, source attestation
- Historical and reconstructed forms

## Notes

- Paradigm numbers classify inflection patterns (important for generation)
- The dictionary covers Sambian and Pomesanian dialects
- Includes historical attestations and modern reconstructions (Neo-Prussian revival)
