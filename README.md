# MetaCare Ember

**The small, open model behind MetaCare Ember, a free visit scribe that runs in a phone's browser, so what is said in
the exam room never leaves the phone.**

[![CI](https://github.com/metacareforyou/metacare-ember/actions/workflows/ci.yml/badge.svg)](https://github.com/metacareforyou/metacare-ember/actions/workflows/ci.yml)
[![Licence: Apache 2.0](https://img.shields.io/badge/licence-Apache%202.0-blue.svg)](LICENSE)

MetaCare Ember records a visit on [Metacare.ai](https://metacare.ai), turns the speech into text on the device, and
drafts the notes, the structured records and the order suggestions for a clinician to review and sign. This repository
makes, trains, scores and exports the model that does one part of that: it reads the visit transcript and writes down
the facts that were said (medicines and doses, results, problems, orders, advice, follow up) in fixed fields, each with
the transcript lines it came from. Fixing misheard drug names, codes and FHIR are plain, tested code in the app.

The model is IBM Granite 4.0 H 350M, fine tuned, and exported to 4 bit ONNX files that run with Transformers.js.
English for now. Code under Apache 2.0; weights and evaluation results will be published openly with a model card.

> Status, November 22, 2026: the first training run is under way. No trained weights are published yet and no
> scores are final.

---

## Where things stand

| Piece | State |
|---|---|
| Visit generator, 16 visit types, every fact checked in the script | Built and tested; first run wrote 2,000 visits, 1,870 kept |
| Synthetic speech and Whisper transcripts of each visit | Built; first run done for 1,870 visits |
| Training pairs, LoRA fine tune, scoring by field | Built; first training run under way |
| Test suite of six recorded visits, with right answers | Built; transcripts by four Whisper models included |
| Export to q4 ONNX for the browser | Built and checked on the base model (see [Export](#export-to-the-browser)) |
| Published weights | None yet |

## How it works

```
 catalog.py      16 visit types; each made up visit gets a spec: speakers, style, and the facts that must be said
      │
 generate.py     an open model writes the dialogue; checks.py keeps it only if every fact is said, the format
      │          holds and no FHIR words leak in (up to 3 attempts)
      │
 speech.py       each turn is spoken by Kokoro 82M in its own voice, with noise and sometimes a phone line, and
      │          transcribed by one of the Whisper models the browser offers
      │
 pairs.py        transcript (written, or as heard) ──► the right answer, taken from the spec, never a model
      │
 finetune.py     LoRA on Granite 4.0 H 350M, then scores held out visit types and the test suite with score.py
      │          (precision and recall for each kind of fact and each field)
      │
 export_onnx.py  the browser's q4 files, size against the iPhone budget, and a smoke generation
```

What matters, and why:

- **The right answers come from specs, not from a model.** A script writer model (Qwen3 14B, Apache 2.0, run with
  llama.cpp on MetaCare's own hardware) only writes the dialogue around facts fixed in advance, and each script is
  checked for every one of them. Writers whose licence limits training other models on their output are not used.
  On one rented L4, 2,000 visits took 157 minutes; 1,870 passed every check.
- **It trains on what Whisper actually hears.** Each visit is also spoken by an open text to speech model and
  transcribed by the browser's Whisper models (tiny, base, small, and large v3 turbo), with room noise, phone lines and
  a few turns given to the wrong speaker, so the model learns to read misheard words ("Levithyroxine", "fura semide").
- **Whole visit types are held out.** Thyroid, migraine and knee follow up visits never appear in training, so the
  score shows whether the model learned to extract rather than to recognise sixteen visits.
- **A separate test suite of six recorded visits** ([`evalsets/suite`](evalsets/suite)), read by stock voices, with
  hand written right answers for every fact and each visit as heard by four Whisper models. Four of the six are visit
  types the training data also has, so their scores are reported apart from the two that are not.
- **The model writes no codes and no FHIR.** It copies what was said into fixed fields. Codes and FHIR resources are
  added by tested code in the app, from code lists confirmed against MetaCare's terminology server, so the model has
  nothing to make up.

## Run it

Python 3.11 or newer. The checks and tests need no GPU:

```bash
git clone https://github.com/metacareforyou/metacare-ember.git
cd metacare-ember
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make check
```

Generation needs an OpenAI compatible server such as llama.cpp's; speech and training need a GPU:

```bash
pip install -e ".[speech,train]"
python -m metacare_ember.generate --out data/private/visits.jsonl --count 2000 \
    --server http://127.0.0.1:8099 --model <model and revision> --parallel 8
python -m metacare_ember.speech --visits data/private/visits.jsonl --out data/private/heard.jsonl
python -m metacare_ember.pairs --visits data/private/visits.jsonl --heard data/private/heard.jsonl \
    --out data/private/pairs.jsonl
python -m metacare_ember.finetune --config configs/granite-h-350m-ember.yaml --pairs data/private/pairs.jsonl --baseline 100
```

Generated visits, transcripts, pairs and weights stay out of the repository (`data/private/` and `runs/` are ignored).

## Export to the browser

```bash
pip install -e ".[export]"
python -m metacare_ember.export_onnx --merged runs/<name>/merged --out runs/<name>/onnx \
    --runtime-from <path to metacare-gpt-frontend>/src/localai/chatManifest.granite-h-350m.json \
    --fp32-check --prompt @<prompt file> --max-new-tokens 200
```

- **`optimum` cannot export Granite 4.0 H** ("a custom or unsupported architecture"). The exporter instead writes the
  merged weights into the graph the app already runs (`onnx-community/granite-4.0-h-350m-ONNX` at commit `06653818`).
  Fine tuning changes weights, not layers.
- **Checked on the base model:** the output is byte for byte the published q4 files; unquantized, it generates the same
  tokens as PyTorch; and Transformers.js 4.3.1 generates the same tokens as Python ONNX Runtime. Not yet checked on a
  trained model, on WebGPU, or on a phone. Details and pinned versions: [docs/EXPORT.md](docs/EXPORT.md).
- **q4 costs accuracy.** On a made up visit the q4 base model drifted from full precision after a few tokens, so the
  model is scored as exported, not only in PyTorch.
- **q4, never q4f16.** Granite 4.0 H in q4f16 answers with a string of exclamation marks on WebGPU (half precision
  overflow).
- **291 MB with the runtime**, against the app's 600 MB iPhone budget.
- The app's manifest is built with `msm manifest` from
  [metacare-small-model](https://github.com/metacareforyou/metacare-small-model), MetaCare's small health question model.

## Layout

```
configs/                  the training config: pinned base model, LoRA, training, export dtype
evalsets/suite/           the six visit test suite: lines, right answers, and Whisper transcripts
docs/EXPORT.md            the export: why, how, versions and measurements
src/metacare_ember/       catalog, checks, generate, speech, pairs, finetune, score, suite, export_onnx
tests/                    run without a GPU or any model download
tools/js-smoke/           checks an export with the Transformers.js version the app ships
```

## Known unknowns

1. **How well 350M parameters extract a visit.** The first scores will say; Granite 4.0 H 1B is the fallback, but it
   needs to fit under the phone budget first.
2. **How fast it runs on phones.** The base 350M model installs on an iPhone and has answered on an iPhone 18; time to
   process a whole visit on a phone has not been measured.
3. **How much q4 costs.** Scores in full precision and as exported will be reported side by side.
4. **Sixteen visit types is narrow.** The held out types test whether that is enough; more types are cheap to add.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Clinicians who want to write or check right answers for test visits are
especially welcome: write to info@metacare.ai.

## Licence and disclaimer

Code: [Apache 2.0](LICENSE). Base model: IBM Granite 4.0, Apache 2.0.

Nothing in this repository, and no output of a model trained with it, is medical advice, diagnosis or treatment.
Ember's notes are drafts for a clinician to review and sign. In an emergency, call 911 or your local emergency number.

Made by [Metacare.ai Inc.](https://metacare.ai) in Miami, Florida.
