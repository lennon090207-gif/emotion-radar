# CLAUDE.md — Emotion Radar project rules

Emotion Radar is **not** a generic scraper. Read this before changing code.

## Core rules

1. **Visual hook analysis matters more than captions.** The caption can lie
   or be unrelated. The frames don't.
2. **Analyze only the first 1–5 seconds.** That's where the hook lives.
   Don't widen the window without a specific reason.
3. **Raw videos are temporary.** Delete MP4s and individual frame JPEGs by
   default after the contact sheet is built. `--keep-temp` is the only
   exception, for debugging.
4. **Store distilled concepts, not large media.** SQLite + a small contact
   sheet per video. No long-term video storage on the VPS.
5. **Track emotional mechanics, not exact hook text.** "Public disrespect
   of an underdog maker" is the asset, not the literal caption.
6. **Generate fresh mutations, not direct copies.** The output should
   suggest *new* hooks that reuse the mechanic in a different niche, not
   restate the source video.
7. **Prioritize freshness/cooked detection.** Being early to a mechanic is
   the entire edge. A cooked mechanic is worthless even if it tested well
   last month.
8. **Keep Apify usage controlled.** Default cap is 3 URLs per command.
   Print cost warning before each run. No scheduled/automated jobs in this
   phase.
9. **Build in phases.** Current phase = URL → contact sheet → report stub.
   Next phase = vision model. Don't skip ahead.
10. **No broad crawling.** No hashtag, profile, search, or related-video
    expansion in this MVP.

## Key fields (schema target)

The report schema tracks both raw metadata and the structured concept:

- **Raw**: platform, video_id, source_url, creator_username,
  creator_nickname, caption, metrics (views/likes/comments/shares/saves),
  duration, cover_url, contact_sheet_path, apify_run_id,
  apify_dataset_id, apify_usage_usd.
- **Visual scene** (filled by vision model later):
  - visual event / what physically happens
  - environment / setting
  - person/people in frame
  - object/product
  - action or conflict
  - on-screen text
- **Concept layer** (filled by vision model later):
  - emotional_mechanic
  - viewer_role (defender, judge, voyeur, learner, accomplice, etc.)
  - emotions_triggered (list)
  - product_attachability (0–1)
  - transferability (0–1, can the mechanic carry to other niches)
  - freshness (0–1, novel right now)
  - cooked_score (0–1, saturation)
  - overall_opportunity_score
  - hook_mutations (list of new hook ideas using the same mechanic)

## Phase boundaries

- This phase: infrastructure only. `analysis.py` returns nulls.
- Next phase: replace `analyze_contact_sheet(...)` with a vision/LLM call
  that reads the contact sheet image and fills the concept fields. Do not
  build that here.
- Later: scoring weights, mutation generation, and (only then) any
  discovery mechanism.

## What NOT to add here

- A web dashboard.
- Hashtag / profile / search scraping.
- Related-video expansion.
- Subtitle/transcript fetching.
- Background schedulers, cron, queue workers.
- Long-term raw video storage.
