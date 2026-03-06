#!/usr/bin/env python3
"""Parse /more/ HTML output from wirdeins.twanksta.org into structured data."""
import sys, re, html


def clean(s):
    """Remove HTML tags, unescape entities, normalize whitespace."""
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    return " ".join(s.split()).strip()


def parse_forms(content):
    """Parse /more/ HTML into structured dict. Used by scraper and CLI."""
    result = {}

    # --- Verb conjugation ---
    cells = re.split(r"<td[^>]*>", content)
    indicative = []
    participles = []

    for cell in cells:
        head_m = re.search(r'<span class="head">(.*?)</span>', cell)
        if not head_m:
            continue
        tense = clean(head_m.group(1))

        if "spoiler-title2" in cell:
            part_m = re.search(r"spoiler-title2.*?<span>(.*?)</span>", cell, re.DOTALL)
            if part_m:
                participles.append({"type": tense, "form": clean(part_m.group(1))})
            continue

        pairs = re.findall(
            r'<span class="pronoun">(.*?)</span>.*?<span class="verb">(.*?)</span>',
            cell, re.DOTALL,
        )
        if pairs:
            indicative.append({
                "tense": tense,
                "forms": [{"pronoun": clean(p), "form": clean(v)} for p, v in pairs],
            })

    if indicative:
        result["indicative"] = indicative

    # Optative
    opt_m = re.search(r'Optative.*?<span class="verb">(.*?)</span>', content, re.DOTALL)
    if opt_m:
        result["optative"] = clean(opt_m.group(1))

    # Imperative
    imp_sec = re.search(r'Imperative.*?(<span class="pronoun.*?)</td>', content, re.DOTALL)
    if imp_sec:
        pairs = re.findall(
            r'<span class="pronoun">(.*?)</span>.*?<span class="verb">(.*?)</span>',
            imp_sec.group(1), re.DOTALL,
        )
        if pairs:
            result["imperative"] = [{"pronoun": clean(p), "form": clean(v)} for p, v in pairs]

    # Subjunctive
    subj_sec = re.search(r'Subjunctive.*?(<span class="pronoun.*?)</td>', content, re.DOTALL)
    if subj_sec:
        pairs = re.findall(
            r'<span class="pronoun">(.*?)</span>.*?<span class="verb">(.*?)</span>',
            subj_sec.group(1), re.DOTALL,
        )
        if pairs:
            result["subjunctive"] = [{"pronoun": clean(p), "form": clean(v)} for p, v in pairs]

    if participles:
        result["participles"] = participles

    # --- Noun/Adjective declension ---
    # Note: width="auto" appears in adjective tables, so match flexibly
    tables = re.findall(r'<table id="subst"[^>]*>(.*?)</table>', content, re.DOTALL)
    declensions = []
    for table_html in tables:
        gender_m = re.search(r'<th class="null">(.*?)</th>', table_html)
        gender = gender_m.group(1) if gender_m else ""

        cases = []
        rows = re.findall(r"<tr>(.*?)</tr>", table_html, re.DOTALL)
        for row in rows:
            case_m = re.search(r'<th class="hea">(.*?)</th>', row)
            if case_m:
                forms = re.findall(r'<span class="verb">(.*?)</span>', row)
                forms = [clean(f) for f in forms]
                cases.append({
                    "case": case_m.group(1),
                    "singular": forms[0] if len(forms) >= 1 else "",
                    "plural": forms[1] if len(forms) >= 2 else "",
                })

        if cases:
            declensions.append({"gender": gender, "cases": cases})

    if declensions:
        result["declension"] = declensions

    # --- Adjective comparison (comparative, superlative) ---
    # Look for ► markers with comparative/superlative forms
    arrows = re.findall(r"►\s*(.*?)</span>", content)
    if arrows:
        comparison_forms = [clean(a) for a in arrows if clean(a)]
        if comparison_forms:
            result["comparison"] = comparison_forms

    # --- Adverb forms ---
    adv_sec = re.search(
        r"Adverb.*?Superlative.*?</tr>(.*?)</table>", content, re.DOTALL
    )
    if adv_sec:
        adv_forms = re.findall(r'<span class="verb">(.*?)</span>', adv_sec.group(1))
        adv_forms = [clean(f) for f in adv_forms if clean(f)]
        if adv_forms:
            labels = ["positive", "comparative", "superlative"]
            result["adverb"] = {
                labels[i]: adv_forms[i] for i in range(min(len(labels), len(adv_forms)))
            }

    return result


def format_output(result):
    """Format parsed forms for CLI display."""
    lines = []

    if "indicative" in result:
        lines.append("=== VERB CONJUGATION ===")
        lines.append("\n  Indicative:")
        for tense_data in result["indicative"]:
            lines.append(f"\n    {tense_data['tense']}:")
            for f in tense_data["forms"]:
                lines.append(f"      {f['pronoun']} {f['form']}")

    if "optative" in result:
        lines.append(f"\n  Optative: {result['optative']}")

    if "imperative" in result:
        lines.append(f"\n  Imperative:")
        for f in result["imperative"]:
            lines.append(f"    {f['pronoun']} {f['form']}")

    if "subjunctive" in result:
        lines.append(f"\n  Subjunctive:")
        for f in result["subjunctive"]:
            lines.append(f"    {f['pronoun']} {f['form']}")

    if "participles" in result:
        lines.append(f"\n  Participles:")
        for p in result["participles"]:
            lines.append(f"    {p['type']}: {p['form']}")

    if "declension" in result:
        lines.append("=== DECLENSION ===")
        for decl in result["declension"]:
            if decl["gender"]:
                lines.append(f"\n  {decl['gender']}:")
            lines.append(f"    {'':15s} {'sing':15s} plur")
            for c in decl["cases"]:
                lines.append(f"    {c['case']:15s} {c['singular']:15s} {c['plural']}")

    if "comparison" in result:
        lines.append(f"\n  Comparison: {' → '.join(result['comparison'])}")

    if "adverb" in result:
        adv = result["adverb"]
        parts = [f"{k}: {v}" for k, v in adv.items()]
        lines.append(f"\n  Adverb: {', '.join(parts)}")

    if not lines:
        lines.append("No forms found.")

    return "\n".join(lines)


if __name__ == "__main__":
    content = sys.stdin.read()
    result = parse_forms(content)
    print(format_output(result))
