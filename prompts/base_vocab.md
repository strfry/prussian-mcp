BASE VOCABULARY (engine-verified — do NOT call search_dictionary or
get_word_forms for anything listed here; the forms below are exact):

Personal pronouns (Nom / Gen / Dat / Acc):

- 1sg: as / māise / mennei / men          1pl: mes / nūse / nūmans / mans
- 2sg: tū / twāise / tebbei / ten         2pl: jūs / jūse / jūmans / wans
- 3sg m: tāns / tenesse / tenesmu / tennan
- 3sg f: tenā / tenesses / tenessei / tennan
- 3pl m: tenēi / tenēisan / tenēimans / tennans
- 3pl f: tennas / tenēisan / tenēimans / tennans
- Possession is expressed with the Gen forms above (māise = mein,
  twāise = dein, nūse = unser, jūse = euer).
- Reflexive dative: sebbei (sich/mir/dir … selbst).

būtwei "to be", indicative present:

- asma (1sg), assei (2sg), ast (3sg AND 3pl), asmai (1pl), astei (2pl)
- For other moods/tenses call get_word_forms("būtwei", features=…).

wīrstwei "to become" (FUTURE auxiliary), indicative present:

- wīrst (1sg/2sg/3sg AND 3pl), wīrstmai (1pl), wīrstei (2pl)
- FUTURE = wīrst(mai/ei) + past active participle of the main verb,
  agreeing in gender/number with the subject:
  get_word_forms("<verb>", features="PrfPrc+Act").
  E.g. as wīrst segīwuns (m) / segīwusi (f) — I will do.
- Invariable short form: the neuter participle, e.g. as wīrst būwus.
- See Syntax rules §11 for the full paradigm.

stas "that; der/die/das" (demonstrative):

- Nom: stas (Sg+Msc), stāi (Sg+Fem; Pl+Msc/Neu), stās (Pl+Fem),
  sta/stan (Sg+Neu)
- Gen: stesse (Sg+Msc/Neu), stesses (Sg+Fem), stēisan (Pl, all genders)
- Dat: stesmu (Sg+Msc/Neu), stessei (Sg+Fem), stēimans (Pl, all genders)
- Acc: stan (Sg, all genders), stans (Pl, all genders)

šis "this; dieser":

- Nom: šis (Sg+Msc), šī (Sg+Fem), šin (Sg+Neu), šāi (Pl+Msc/Neu),
  šās (Pl+Fem)
- Gen: šisse (Sg+Msc/Neu), šisses (Sg+Fem), šēisan (Pl)
- Dat: šismu (Sg+Msc/Neu), šissei (Sg+Fem), šēimans (Pl)
- Acc: šin (Sg+Msc/Neu), šan (Sg+Fem), šins (Pl+Msc/Neu), šans (Pl+Fem)

aīns "one" (numeral):

- Nom: aīns (Sg+Msc), ainā (Sg+Fem), aīnan (Sg+Neu)
- Gen: ainasse (Sg+Msc), ainasses (Sg+Fem)
- Dat: ainasmu (Sg+Msc), ainassei (Sg+Fem)
- Acc: aīnan (Sg, all genders)
- German indefinite article "ein/eine" is normally NOT translated;
  use aīns only when it means the numeral "one".

Question/relative pronoun: kas (wer; der/die rel., Nom+Sg+Msc) — for
other cases call get_word_forms("kas", …).

Prepositions with government (case is fixed — never guess):

- ēn + Acc (in — direction) / + Dat (in — location)
- sēn + Acc (mit)
- prēi + Acc (zu, bei, an)
- iz + Acc (aus, von)
- pēr + Acc (für)
- nō + Acc (auf)
- pa + Dat (unter) / + Acc (nach, gemäß)
- kīrsa + Acc (über, quer über)
- be + Acc (ohne); also sklāit (außer, ohne)

Particles and conjunctions:

- be = und (conjunction — same surface form as the preposition "ohne";
  attested orthography "bhe")
- adder = oder; aber
- kāi = dass, damit, um zu (takes SUBJUNCTIVE — see main prompt)
- ni = nicht, kein, weder

High-frequency verb LEMMAS (skip search_dictionary, go straight to
get_word_forms with these):

- turītun = haben; also müssen/sollen
- mazītwei = können; warītun = imstande sein
- pastātwei = werden (see wīrstwei above for the FUTURE auxiliary use)
- ēitwei = gehen; perēitwei = kommen
- segītun = tun; tikīntun = machen, herstellen
- "wollen" has NO full verb: only defective kwāi (1/2sg present) and
  the noun kwāits (Wille) exist — rephrase or flag as uncertain.
