# Emotion Radar

Private tool for organic marketing research. Detects fresh emotional hook
concepts in the **first 1–5 seconds** of short-form video (TikTok first).

This is NOT a generic scraper. The goal is to understand the *visual* opening
hook of a video — what you see, what text is on screen, what emotional
mechanic is being used, and whether the concept is fresh or already cooked.

## Why visual hooks

Captions and TikTok page text often hide the real hook. Example: an HTTYD
lamp video whose caption was *"Please be honest, how are they?"* — but the
actual hook was a maker at a market stall watching a stranger pick up his
lamp and smash it on the floor. The mechanic (public disrespect +
underdog maker + viewer-defense instinct) lives in the frames, not the
caption. So the analysis pipeline runs on frames extracted from the
first 5 seconds, not on metadata text.

## What this tool does

```
TikTok URL
  -> Apify (clockworks/tiktok-video-scraper)
  -> metadata + downloadable MP4
  -> temporary download to data/tmp/videos/
  -> ffmpeg extracts dense frames in the first 5s
  -> compressed contact sheet at data/contact_sheets/{video_id}.jpg
  -> two-pass vision analysis:
       Pass 1 (Visual Event Extractor)  -- literal chronological evidence
       Pass 2 (Hook Strategist)         -- mechanic + 6 mutations
  -> structured report saved to SQLite
  -> raw MP4 and individual frames deleted
  -> (if video_id matches a known fixture) calibration check
```

The single command `analyze-link URL` runs the whole pipeline end-to-end.
`analyze-url` runs only the infrastructure half (no vision); `analyze-report`
re-runs the two-pass analysis on an existing report.

## Setup

### 1. ffmpeg

Install ffmpeg and make sure `ffmpeg` is on `PATH`.

```bash
# Debian/Ubuntu (VPS)
apt-get update && apt-get install -y ffmpeg
```

### 2. Python deps

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. APIFY_TOKEN

The client reads `APIFY_TOKEN` from:

1. process environment, or
2. `/root/.hermes/.env` (VPS path), or
3. `./.env` (local dev fallback).

`.env.example` shows the expected format. Never commit `.env`.

### 4. Vision API key (Phase 2)

`analyze-report` needs a vision-capable LLM. Set ONE of:

- `OPENAI_API_KEY` — uses `https://api.openai.com/v1` by default.
- `OPENROUTER_API_KEY` — uses `https://openrouter.ai/api/v1` by default.

Optional overrides:

- `OPENAI_BASE_URL` — any OpenAI-compatible endpoint (OpenRouter, vLLM, etc.).
- `VISION_MODEL` — model used by both passes when no per-pass override is set. Defaults to `gpt-4o`.
- `VISION_EVENT_MODEL` — Pass 1 (visual event extraction). Falls back to `VISION_MODEL`, then `gpt-4o`. Use this slot for your strongest vision model.
- `HOOK_STRATEGY_MODEL` — Pass 2 (hook strategy + mutations). Text-only call; safe to point at a cheaper model. Falls back to `VISION_MODEL`, then `gpt-4o`.

Same resolution order as `APIFY_TOKEN` (env -> `/root/.hermes/.env` -> `./.env`). Keys are never logged or printed.

### Recommended models (from real VPS testing on the Oliver canary)

```
VISION_EVENT_MODEL=google/gemini-2.5-flash
HOOK_STRATEGY_MODEL=gpt-4o
```

On the Oliver HTTYD-lamp video, `gpt-4o` and `claude-sonnet-4.5` both
missed the actual physical event (someone destroying the lamp) and
softened it into mood. `google/gemini-2.5-flash` correctly identified
the object motion across the dense contact sheet — "man picks up a
lamp, drops it, and the lamp breaks on the ground" — so it's the
current recommendation for **Pass 1**. The Pass-2 prompt has explicit
rules forcing it not to soften Pass 1's evidence, but Pass 2's job is
text reasoning, so any reasonably strong text model (gpt-4o, claude,
gemini) works for that slot.

This recommendation will go stale as models improve. The reliable test
is the calibration canary itself (see "Calibration check" below).

## Commands

```bash
# ONE-SHOT (TikTok URL): everything end-to-end (Apify -> contact sheet -> three-pass vision -> report -> auto-calibration)
python -m emotion_radar analyze-link https://www.tiktok.com/@user/video/123
python -m emotion_radar analyze-link URL --dry-run-vision   # build contact sheet, print prompts, no API call
python -m emotion_radar analyze-link URL --no-vision        # stop at contact sheet + report stub
python -m emotion_radar analyze-link URL --no-specificity   # only Pass 1 + Pass 2 (skip Phase-6 specificity)
python -m emotion_radar analyze-link URL --skip-evaluation  # disable auto-calibration on known fixtures
python -m emotion_radar analyze-link URL --expected SPEC.json  # custom calibration spec
python -m emotion_radar analyze-link URL --keep-temp        # keep raw mp4/frames for debugging

# LOCAL SEED CLIPS (Phase 7): same three-pass pipeline, no Apify
python -m emotion_radar analyze-file /path/to/clip.mp4
python -m emotion_radar analyze-folder /path/to/folder --limit 10
python -m emotion_radar analyze-folder /path/to/folder --recursive --limit 20

# infrastructure-only (no vision): Apify -> contact sheet -> stub report
python -m emotion_radar analyze-url https://www.tiktok.com/@user/video/123

# batch infrastructure (one URL per line, # for comments, blanks ignored)
python -m emotion_radar analyze-urls urls.txt

# re-run two-pass vision on an existing report
python -m emotion_radar analyze-report REPORT_ID
python -m emotion_radar analyze-report REPORT_ID --dry-run   # print both pass prompts, no API call

# calibration check against a known-hook spec (case-insensitive substring)
python -m emotion_radar evaluate-report REPORT_ID --expected docs/examples/oliver_expected.json

# list / show / cleanup
python -m emotion_radar list-reports
python -m emotion_radar show-report REPORT_ID
python -m emotion_radar cleanup-temp
```

### Two-pass analysis

`analyze-link` and `analyze-report` use two LLM passes:

| Pass | Role | Input | Output |
|------|------|-------|--------|
| 1 — Visual Event Extractor | Vision | The contact sheet image | Literal, chronological frame observations + physical action + conflict type + confidence. NO scoring, NO hook ideas. |
| 2 — Hook Strategist | Text-only | Pass 1's JSON + video metadata | Viral mechanic, scroll-stop reason, viewer role, comment/share triggers, cooked elements, virality scores, and exactly 8 **creative hook concepts** (2 same_mechanic / 3 adjacent_leap / 2 big_swing / 1 wildcard). The product is secondary — the concepts mutate the emotional situation, not the object. |

Why two passes: in one-pass mode the model tended to short-circuit from
"market stall" straight to "creator looks discouraged" and miss the
actual physical action (someone smashing the product). Forcing Pass 1 to
write the chronological evidence record before any strategy reasoning
makes that failure mode much harder.

`raw_analysis` in the SQLite row carries both passes verbatim
(`analysis_mode: "two_pass"`, `visual_event_pass: {...}`,
`hook_strategy_pass: {...}`) so you can audit either pass after the fact.

### Output mental model (Phase 4)

The tool answers six questions about every video:

1. What is the **viral mechanic**?
2. Why does it **stop the scroll** in the first 1–2 seconds?
3. What **viewer role** does it create?
4. What is the **comment trigger** / **share trigger**?
5. Which **parts are cooked** right now?
6. What are 8 **broad creative hook concepts** that re-use the same mechanic in NEW emotional situations?

The product/niche of the source video is **not** the asset. The mechanic
is. The concepts are intentionally broad — they're meant to attach to
many products later, not just the one in the source video. Two concepts
that only differ in what object is on screen count as one concept.

### Local seed clip ingestion (Phase 7)

When you already have a folder of curated viral clips (e.g. a public
Google Drive folder of known-viral hooks), use `analyze-file` /
`analyze-folder` instead of `analyze-link`. They skip Apify entirely.

Each seed-clip report carries this in `raw_analysis.source_metadata`:

```json
{
  "source_type": "drive_seed_clip",
  "source_filename": "lobster_bag_drop.mp4",
  "known_viral": true,
  "analytics_available": false,
  "original_local_path": "/abs/path/to/lobster_bag_drop.mp4"
}
```

Supported extensions: `.mp4 .mov .m4v .webm`.

#### Pulling a public Drive folder with gdown

```bash
pip install gdown

# Public folder URL goes here. --remaining-ok lets gdown skip files it
# can't fetch (e.g. items requiring auth) rather than aborting.
gdown --folder "https://drive.google.com/drive/folders/<FOLDER_ID>" \
  -O data/imports/friend_drive --remaining-ok

python -m emotion_radar analyze-folder data/imports/friend_drive --limit 10
```

#### analyze-folder behavior

- **Non-recursive by default.** Use `--recursive` to walk subdirectories.
- **Sorted by filename** for deterministic order.
- **Default `--limit 5`** to avoid surprise vision-API cost. Raising the
  limit prints a clear warning but is allowed.
- **Continues on per-file failure** — one bad clip won't kill the batch.
- **Batch summary** at the end: analyzed/failed counts, report ids,
  dominant story flows, viral mechanics per file, failures.
- **Does NOT auto-run calibration** per file. Seed clips don't have
  analytics, and the Oliver canary only fires on a specific TikTok
  video_id that local files won't match.

### Calibration / regression canary

`evaluate-report` reads a small JSON file describing what a *correct*
analysis of a known video should mention. It's a substring check, not a
semantic judge — it exists to catch the failure mode where the model
hallucinates a generic "creator looks discouraged" reading and misses
the actual visual hook (someone smashing the product, etc.).

Spec shape (`docs/examples/oliver_expected.json` is the working
example):

```json
{
  "required_terms": ["market stall"],

  "required_any": [
    ["smashed", "thrown", "dropped", "broken", "shattered", "on the ground"],
    ["dragon lamp", "HTTYD lamp", "Dragon Resin Lamp"],
    ["public disrespect", "rejection", "mockery", "insult"],
    ["underdog maker", "handmade seller", "small creator"]
  ],

  "forbidden_terms": ["street musician", "SaaS", "crypto"],

  "expected_mechanic": "public disrespect + underdog maker",
  "mechanic_any": [
    "public disrespect + underdog maker",
    "public rejection of underdog maker",
    "viewer-defense"
  ]
}
```

- `required_terms`: every entry must appear in the report's text.
- `required_any`: each group must have at least one entry that appears.
  Use this for word choice the model can legitimately vary
  (`dropped` vs `smashed` vs `thrown`, `dragon lamp` vs `HTTYD lamp`, etc.).
- `forbidden_terms`: any hit fails the calibration (taste regression).
- `expected_mechanic` / `mechanic_any`: the report's `emotional_mechanic`
  must match at least one of these (case-insensitive substring, either
  direction).

Non-zero exit code on failure, so it can run in CI or a shell pipeline.

### Flags

- `--keep-temp` — keep the raw MP4 and individual frames after the contact
  sheet is built. Default is to delete them.
- `--confirm-large` — required to analyze more than 3 URLs in one command.
  Default cap is 3 (Apify cost safety).
- `--db PATH` — override SQLite path. Default `data/emotion_radar.db`.

## Storage behavior

| Artifact            | Location                              | Kept?         |
|---------------------|---------------------------------------|---------------|
| Raw MP4             | `data/tmp/videos/{video_id}.mp4`      | Deleted       |
| Individual frames   | `data/tmp/frames/{video_id}/*.jpg`    | Deleted       |
| Contact sheet       | `data/contact_sheets/{video_id}.jpg`  | **Kept**      |
| Report row + JSON   | `data/emotion_radar.db`               | **Kept**      |

`--keep-temp` overrides the deletion. `cleanup-temp` removes anything
still left under `data/tmp/`.

## Apify cost warning

`clockworks/tiktok-video-scraper` charges per video. Roughly $0.025 per 3
videos in testing. Each `analyze-url` / `analyze-urls` invocation prints
how many URLs will be sent before calling Apify. The default 3-URL cap is
a deliberate safety net; raise it with `--confirm-large` only when you
mean it.

This MVP intentionally does **not** schedule automated runs, scrape
profiles/hashtags/search, fetch related videos, or download subtitles.

## Manual test

These three URLs were used during build/validation:

```
https://www.tiktok.com/@olivermakesartt/video/7623559389307211030
https://www.tiktok.com/@jaydenshells2/video/7633803038083337503
https://www.tiktok.com/@clayheartco/video/7618847873228180757
```

Start with the first one only:

```bash
python -m emotion_radar analyze-link https://www.tiktok.com/@olivermakesartt/video/7623559389307211030
```

The Oliver video id (`7623559389307211030`) is wired to
`docs/examples/oliver_expected.json` as a known calibration fixture, so
`analyze-link` auto-runs the canary at the end of the run. If the
calibration fails, the CLI prints `Calibration failed. Do not trust this
report yet.` and the report is preserved so you can inspect it.

## Tests

```bash
pytest
```

Tests do not hit Apify. Network calls are mocked.
