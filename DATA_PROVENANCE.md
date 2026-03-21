# 📚 Daten-Provenance: Die 975 Einträge ohne Übersetzung

**Status:** Absichtlich im Corpus - Dies sind historische Referenzen!

---

## 🔍 Was wir entdeckt haben:

**ALL 975 Einträge ohne Übersetzung haben LINGUISTISCHE REFERENZEN im `desc` Feld!**

```
"paauginnaiti" → desc: "↑ Paaugīntun ip 2 sg m [poauginneiti 93]"
"paaugīnts"   → desc: "↑ Paaugīntun pc pt pa [poaugints 133]"
"abbeimas"    → desc: "↑ Abbai dat"
"wābelka"     → desc: "↑ Wābli dm [Wobelke Gr]"
"ainā"        → desc: "↑ Aīns f [ainā 61(21)]"
```

---

## 💡 Was bedeuten diese Notationen?

### Format: `↑ REFERENZ grammatik [quelle seitennummer]`

#### Beispiel 1: `↑ Paaugīntun ip 2 sg m [poauginneiti 93]`
```
↑                  = "See also / Verwandte Wortform"
Paaugīntun         = Lemma (Wörterbuch-Eintrag zum Nachschlagen)
ip 2 sg m          = Grammatikalische Kategorie
                     ip = ?
                     2  = 2. Person
                     sg = Singular
                     m  = Maskulinum
[poauginneiti 93]  = Quellenangabe: "Werk: poauginneiti, Seite 93"
```

#### Beispiel 2: `↑ Abbai dat`
```
↑       = Verwandter Eintrag
Abbai   = Nominativ (Lemma)
dat     = Diese Form ist der Dativ
```

#### Beispiel 3: `[Wobelke Gr]`
```
[Wobelke Gr]  = Quellenangabe aus Grammatik (Gr) von "Wobelke"
```

---

## 📖 Root Cause: Historisches Palmaitis Corpus

Diese Einträge sind **Inflektionsformen & Varianten** aus dem **Palmaitis Originalkorpus**, die:

1. **Im Original dokumentiert** - Mit exakten Quellenreferenzen
2. **Aber ohne Übersetzung** - Weil die Semantik aus dem Lemma (mit `↑` referenziert) klar ist
3. **Mit Quellenangaben** - [seitennummer] verweist auf Originalquelle
4. **Linguistisch wertvoll** - Zeigt Flexion, Variation, historische Formen

Beispiel:
```
Lemma:     "paaugintun"  (mit Übersetzung: "ankündigen")
Form:      "paauginnaiti" (ohne Übersetzung - aber `↑ Paaugīntun` verweist darauf!)
           → 2. Person Singular Maskulinum Form dieses Verbs
```

---

## ✅ WICHTIG ZU DOKUMENTIEREN:

Diese 975 Einträge sind **FEATURE, NICHT BUG!**

Sie dokumentieren:
- ✓ Historische Wortformen aus Palmaitis
- ✓ Inflektionale Variation
- ✓ Quellenbelege mit Seitennummern
- ✓ Grammatikalische Kategorien
- ✓ Wissenschaftliche Integrität der Datensammlung

---

## 🎯 Praktische Konsequenzen:

### Für den Chatbot:

```javascript
lookup("paauginnaiti")
→ { 
    word: "paauginnaiti",
    desc: "↑ Paaugīntun ip 2 sg m [poauginneiti 93]"
    translations: {}  // ← ABSICHTLICH LEER
  }

// LLM erhält:
"Der Benutzer fragt nach 'paauginnaiti'.
Das ist eine Wortform von 'Paaugīntun' (2. Sg. Mask.)
Siehe Quelle: poauginneiti Seite 93"

// Besser als: Zu ignorieren oder als "Fehler" zu behandeln!
```

### Empfehlung:

**Diese Einträge sollten im API-Response gekennzeichnet sein:**

```javascript
{
  word: "paauginnaiti",
  isForm: true,  // ← Neue Flag
  baseForm: "Paaugīntun",  // ← Aus desc extrahiert
  grammaticalInfo: "2 sg m",  // ← Aus desc extrahiert
  source: "poauginneiti 93",  // ← Aus desc extrahiert
  desc: "↑ Paaugīntun ip 2 sg m [poauginneiti 93]",
  translations: {}  // ← Absichtlich - siehe baseForm
}
```

---

## 📋 Daten-Qualitäts-Klassifizierung

```
Einträge mit Übersetzung:        9.197  (90.4%)  ✅ COMPLETE
Einträge = Wortformen/Referenzen:  975  (9.6%)   ✅ HISTORICAL DATA
────────────────────────────────────────────────
GESAMT GÜLTIG:                 10.172  (100%)    ✅ INTACT
```

Diese Klassifizierung sollte **im Data Codebook dokumentiert** werden!

---

## 🔧 Verbesserungs-Möglichkeiten:

### Phase 1 (Dokumentation):
- ✓ Erklären, dass diese 975 sind historische Referenzen
- ✓ Dokumentieren im README / DATA_CODEBOOK
- ✓ Nicht als "fehlende Daten" behandeln

### Phase 2 (Strukturierung):
- [ ] `desc` Feld mit Regex parsen
- [ ] `baseForm`, `grammaticalInfo`, `source` extrahieren
- [ ] Neue Felder im JSON-Schema definieren

### Phase 3 (API/LLM):
- [ ] API: Unterscheidung `isForm: true/false`
- [ ] LLM Prompt: "Wenn `isForm: true`, gib Benutzer `baseForm` und Grammatik"
- [ ] Chatbot: Bessere UI für Flexionsformen

---

## 📚 Beispiel-Pattern aus `desc`:

```regex
# Pattern: "↑ LEMMA grammatik [quelle]"
↑\s+([A-Zā-ž]+)\s+([a-z0-9\s]+)\s+\[(.+?)\]

# Extraktion:
Group 1 = Lemma-Referenz ("Paaugīntun")
Group 2 = Grammatik ("ip 2 sg m")
Group 3 = Quelle ("poauginneiti 93")
```

Diese könnten **automatisiert strukturiert** werden!

---

## ✨ Konklusion:

**Die 975 "fehlenden" Übersetzungen sind nicht fehlend - sie sind wissenschaftliche Referenzen!**

Dies ist eine **Stärke des Corpus**, nicht eine Schwäche:
- Zeigt Quellenbelege
- Dokumentiert historische Variation
- Ermöglicht linguistische Forschung
- Bewahrt akademische Integrität

**Empfehlung:** Nicht "fixen", sondern **anders dokumentieren und strukturieren**!

---

Generated: 2026-03-06 GMT+1
Author: Data Analysis via Automated Testing
