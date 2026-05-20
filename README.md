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

## What this MVP does

```
TikTok URL
  → Apify (clockworks/tiktok-video-scraper)
  → metadata + downloadable MP4
  → temporary download to data/tmp/videos/
  → ffmpeg extracts frames at t=0,0.5,1,1.5,2,3,4,5s
  → compressed contact sheet at data/contact_sheets/{video_id}.jpg
  → structured report saved to SQLite
  → raw MP4 and individual frames deleted
```

The vision/LLM analysis is a placeholder in this phase. The schema is
ready for it; wiring it up is the next phase.

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

- `OPENAI_BASE_URL` — point at any OpenAI-compatible endpoint.
- `VISION_MODEL` — defaults to `gpt-4o`. Set to whatever vision-capable model your provider exposes.

Same resolution order as `APIFY_TOKEN` (env → `/root/.hermes/.env` → `./.env`). Keys are never logged or printed.

## Commands

```bash
# analyze a single URL
python -m emotion_radar analyze-url https://www.tiktok.com/@user/video/123

# analyze multiple URLs (one per line, # for comments, blanks ignored)
python -m emotion_radar analyze-urls urls.txt

# list saved reports (id, platform, creator, caption snippet, views, score, time)
python -m emotion_radar list-reports

# show one report as pretty JSON
python -m emotion_radar show-report REPORT_ID

# run vision analysis on an existing report's contact sheet
python -m emotion_radar analyze-report REPORT_ID
python -m emotion_radar analyze-report REPORT_ID --dry-run   # print prompt only, no API call

# delete old temp videos and frames (contact sheets are preserved)
python -m emotion_radar cleanup-temp
```

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
python -m emotion_radar analyze-url https://www.tiktok.com/@olivermakesartt/video/7623559389307211030
```

## Tests

```bash
pytest
```

Tests do not hit Apify. Network calls are mocked.
