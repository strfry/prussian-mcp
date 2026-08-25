BASE VOCABULARY (engine-verified — do NOT call search_dictionary or
get_word_forms for anything listed here; the forms below are exact):

Personal pronouns (Nom / Gen / Dat / Akk):

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
  get_word_forms("<verb>", features="Part+Pret").
  E.g. as wīrst segīwuns (m) / segīwusi (f) — I will do.
- Invariable short form: the neuter participle, e.g. as wīrst būwus.
- See Syntax rules §11 for the full paradigm.

stas "that; der/die/das" (demonstrative):

- Nom: stas (Sg+Masc), stāi (Sg+Fem; Pl+Masc/Neut), stās (Pl+Fem),
  sta/stan (Sg+Neut)
- Gen: stesse (Sg+Masc/Neut), stesses (Sg+Fem), stēisan (Pl, all genders)
- Dat: stesmu (Sg+Masc/Neut), stessei (Sg+Fem), stēimans (Pl, all genders)
- Akk: stan (Sg, all genders), stans (Pl, all genders)

šis "this; dieser":

- Nom: šis (Sg+Masc), šī (Sg+Fem), šin (Sg+Neut), šāi (Pl+Masc/Neut),
  šās (Pl+Fem)
- Gen: šisse (Sg+Masc/Neut), šisses (Sg+Fem), šēisan (Pl)
- Dat: šismu (Sg+Masc/Neut), šissei (Sg+Fem), šēimans (Pl)
- Akk: šin (Sg+Masc/Neut), šan (Sg+Fem), šins (Pl+Masc/Neut), šans (Pl+Fem)

aīns "one" (numeral):

- Nom: aīns (Sg+Masc), ainā (Sg+Fem), aīnan (Sg+Neut)
- Gen: ainasse (Sg+Masc), ainasses (Sg+Fem)
- Dat: ainasmu (Sg+Masc), ainassei (Sg+Fem)
- Akk: aīnan (Sg, all genders)
- German indefinite article "ein/eine" is normally NOT translated;
  use aīns only when it means the numeral "one".

Question/relative pronoun: kas (wer; der/die rel., Nom+Sg+Masc) — for
other cases call get_word_forms("kas", …).

Prepositions with government (case is fixed — never guess):

- ēn + Akk (in — direction) / + Dat (in — location)
- sēn + Akk (mit)
- prēi + Akk (zu, bei, an)
- iz + Akk (aus, von)
- pēr + Akk (für)
- nō + Akk (auf)
- pa + Dat (unter) / + Akk (nach, gemäß)
- kīrsa + Akk (über, quer über)
- be + Akk (ohne); also sklāit (außer, ohne)

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
