# ML Investing v2

Denne versjonen er laget for å teste om et aksjesignal kan overleve realistisk
utførelse og kostnader. Originalfilene i `sources/` er urørt.

## Hva som er endret

- Signalet beregnes etter sluttkurs på dag `t` og kan først handles på justert
  åpning neste børsdag.
- Forward-labelen bruker samme open-to-open-periode som backtesten.
- Kandidatlisten bestemmes uten å filtrere på om fremtidig avkastning finnes.
- Punkt-i-tid-univers støttes med `listed_from` og `listed_to`.
- Modellen refittes walk-forward og ser bare etiketter med `label_date` før
  refit-datoen.
- Modeller velges på historisk rank-IC. Bare modeller med minst 0,01 i
  validerings-IC inngår i ensemblet; uten validert edge beholdes kontanter.
- Modellen lærer absolutt open-to-open-avkastning. Nivået i prognosen kalibreres
  på en eldre valideringsperiode og krympes mot historisk gjennomsnitt.
- Porteføljen har rangeringsbuffer, inverse-volatilitetsvekter, maksvekt,
  no-trade-bånd og en terskel som må dekke risikofri avkastning, kurtasje,
  spread og forventet impact for hele rundturen.
- Backtesten fører kontanter og posisjoner daglig og inkluderer kurtasje,
  spread, kvadratrot-impact, volumkapasitet, delisting og konservativ behandling
  av stale priser.
- ML-resultatet sammenlignes med 12–1 momentum under samme utførelse og
  kostnader. Momentumets forventede avkastning kalibreres bare på utfall som
  var kjent før hver signaldato.

## Prisdata

Hver CSV-rad må være én unik `(ticker, date)` og ha:

```text
ticker,date,open,close,adj_close,volume
```

`high` og `low` kan være med. Følgende valgfrie felt støttes:

- `exchange`: valgfri kalender-ID. Én kjøring kan bare inneholde én børs;
  kjør børser separat slik at neste åpning og fridager modelleres riktig.
- `in_universe`: boolsk punkt-i-tid-medlemskap.
- `delisting_return`: inkrementell avkastning etter siste dags `adj_close`.
  Delisting-raden må være instrumentets siste rad. Tapet brukes både i
  treningsetiketten og backtesten.

Prisfilene må dekke alle faktiske sesjoner for den aktuelle børsen. Programmet
kan oppdage inkonsistente eller manglende enkeltaksjer, men kan ikke vite at en
hel børsdag mangler fra alle filene uten en ekstern børskalender.

Nordiske priser, omsetning og porteføljeverdi må være konvertert til samme
basisvaluta før kjøring. Ellers er likviditetsgrenser og avkastning ikke
sammenlignbare.

Et separat univers kan leveres som:

```text
ticker,listed_from,listed_to
AAA,2004-05-01,2021-09-30
BBB,2010-03-15,
```

Uten `in_universe` eller en slik fil kan ikke simuleringen anses som trygg mot
survivorship bias. Historikken må også inneholde avnoterte selskaper og korrekte
delisting-utbetalinger; medlemskapsfilen alene kan ikke gjenopprette manglende
prisdata.

Backtestens signalgrensesnitt er bevisst strengt: hver rad må inneholde `date`,
`ticker`, dimensjonsløs `score`, absolutt `expected_return`,
`horizon_sessions` og `model_refit_date`. Backtesten avviser en modell som er
refittet etter signaldatoen, og bruker aldri en rangeringsscore som om den var
en prosentavkastning.

## Kjøring

Til vanlig trenger du bare å oppdatere prisene og kjøre modellen:

```bash
python ../src/fetch_prices.py
python model.py
```

`model.py` kjører både den realistiske historiske simuleringen og dagens
aksjevalg. Den finner pris- og resultatmappene automatisk. Ugyldige gamle
leverandørrader ignoreres uten at råfilene endres.

Standardstrategien holder fem aksjer. Rangeringen kombinerer 25 % walk-forward
ML og 75 % 12–1 momentum, valgt på utviklingsperioden 2020–2023 og kontrollert
separat på perioden fra 2024. Porteføljen bruker en rangeringsbuffer på åtte
navn, men eier aldri mer enn fem samtidig.

Installer den lokale pakken:

```bash
python -m pip install -e .
```

Kjør en forhåndsdefinert walk-forward-periode:

```bash
mlinvesting research \
  --prices-dir data/raw_prices \
  --universe data/universe_history.csv \
  --start 2020-01-02 \
  --capital 1000000 \
  --top-n 10 \
  --output-dir output/v2
```

Generer siste kandidatliste, refittet på alle kjente etiketter. Kommandoen viser
bare navn som dekker risikofri avkastning og de angitte kostnadene:

```bash
mlinvesting latest \
  --prices-dir data/raw_prices \
  --universe data/universe_history.csv \
  --top-n 10 \
  --output-dir output/v2
```

Kjør testpakken:

```bash
python -m pytest -q
```

## Forskningsdisiplin

`--start`, univers, holdeperiode, funksjonssett og kostnadsantakelser bør låses
før resultatet ses. Hvis de endres etterpå, er perioden utviklingsdata og kan
ikke omtales som en urørt test. Endelig vurdering bør gjøres med en ny lockbox
eller paper trading.

Høyere backtestavkastning er ikke en garanti for høyere virkelig avkastning.
Målet med v2 er først å fjerne kjente optimistiske skjevheter og deretter finne
signaler som gir positiv nettoavkastning med realistisk kapasitet. Det følger
ingen historiske prisfiler med denne speilingen, så reell avkastning må måles på
et komplett punkt-i-tid-datasett før strategien vurderes for paper trading.
