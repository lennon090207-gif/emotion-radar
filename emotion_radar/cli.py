"""CLI entrypoints.

Commands:
  analyze-link URL                 — one-shot: Apify -> contact sheet -> two-pass vision -> report
  analyze-url URL                  — infrastructure-only: stops at contact sheet + report stub
  analyze-urls FILE                — multi-URL infrastructure-only batch
  analyze-report REPORT_ID         — re-run two-pass vision on an existing report
  evaluate-report REPORT_ID        — calibration check against expected.json
  list-reports                     — show recent reports
  show-report REPORT_ID            — pretty-print one report as JSON
  cleanup-temp                     — delete temp videos and frames

Global flags:
  --keep-temp                      — don't delete raw mp4/frames after the sheet
  --confirm-large                  — required for > 3 URLs in one command
  --db PATH                        — override SQLite path
  --data-dir PATH                  — override data root
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click

from . import analysis as analysis_mod
from .apify_client import ApifyClient, ApifyError, normalize_items
from .cleanup import (
    cleanup_temp as cleanup_temp_dirs,
    remove_frame_dir,
    remove_video_file,
)
from .config import (
    DEFAULT_MAX_URLS,
    FRAME_TIMESTAMPS_SEC,
    enforce_url_cap,
    get_apify_token,
    load_env,
    resolve_paths,
)
from .db import (
    ALLOWED_RATINGS,
    FeedbackError,
    build_taste_summary,
    get_report,
    get_report_feedback,
    insert_feedback,
    insert_report,
    list_feedback,
    list_reports,
    update_report_analysis,
)
from .evaluator import (
    EvaluationResult,
    evaluate_report as evaluate_report_fn,
    load_expected,
)
from .models import NormalizedItem
from .providers import (
    ROLE_HOOK_STRATEGY,
    ROLE_VISION_EVENT,
    VisionProvider,
    VisionProviderError,
    build_provider_for_role,
)
from .video import build_contact_sheet, download_video, extract_frames, VideoError


# Known-fixture mapping: video_id -> calibration spec path (relative to repo).
# analyze-link auto-runs the spec when the analyzed video matches.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWN_FIXTURES: dict[str, Path] = {
    "7623559389307211030": _PROJECT_ROOT / "docs" / "examples" / "oliver_expected.json",
}

# Phase 7: local seed-clip ingestion.
SUPPORTED_VIDEO_EXTS: frozenset[str] = frozenset({".mp4", ".mov", ".m4v", ".webm"})
DEFAULT_FOLDER_LIMIT = 5


# ---- helpers ---------------------------------------------------------------

def _read_url_file(path: Path) -> list[str]:
    if not path.is_file():
        raise click.ClickException(f"URL file not found: {path}")
    urls: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    if not urls:
        raise click.ClickException(f"No URLs found in {path}.")
    return urls


def _truncate(s: str | None, n: int) -> str:
    if not s:
        return ""
    s = s.replace("\n", " ").replace("\r", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _process_one(
    item: NormalizedItem,
    paths,
    run_info,
    keep_temp: bool,
    apify_token: str | None = None,
) -> dict[str, Any]:
    """Process a single normalized item through download -> frames ->
    contact sheet -> analysis stub -> report dict. Errors in any step are
    captured and returned as report.error rather than raised, so a batch
    can continue."""
    report: dict[str, Any] = {
        "platform": item.platform,
        "source_url": item.source_url,
        "submitted_url": item.submitted_url,
        "video_id": item.video_id,
        "creator_username": item.creator_username,
        "creator_nickname": item.creator_nickname,
        "caption": item.caption,
        "metrics": item.metrics,
        "duration": item.duration,
        "cover_url": item.cover_url,
        "video_download_url_saved": False,
        "apify_run_id": run_info.run_id,
        "apify_dataset_id": run_info.dataset_id,
        "apify_usage_usd": run_info.usage_total_usd,
        "apify_charged_events": run_info.charged_events,
        "contact_sheet_path": None,
        "error": item.error,
    }
    if item.error:
        return report
    if not item.video_download_url:
        report["error"] = "No video_download_url on normalized item."
        return report

    video_id = item.video_id or "unknown"
    video_path: Path | None = None
    frames_dir = paths.tmp_frames_dir / video_id
    # Apify key-value-store URLs need the Authorization header; public CDN
    # URLs (rare path) do not. Send the bearer only for api.apify.com hosts.
    dl_headers: dict[str, str] | None = None
    if apify_token and "api.apify.com" in item.video_download_url:
        dl_headers = {"Authorization": f"Bearer {apify_token}"}
    try:
        click.echo(f"  -> downloading video for {video_id} ...")
        video_path = download_video(
            item.video_download_url,
            paths.tmp_videos_dir,
            video_id,
            headers=dl_headers,
        )
        click.echo(f"  -> extracting frames at {list(FRAME_TIMESTAMPS_SEC)}s ...")
        frame_paths = extract_frames(video_path, frames_dir, FRAME_TIMESTAMPS_SEC)
        sheet_path = paths.contact_sheets_dir / f"{video_id}.jpg"
        click.echo(f"  -> building contact sheet -> {sheet_path}")
        build_contact_sheet(
            frame_paths,
            list(FRAME_TIMESTAMPS_SEC),
            sheet_path,
        )
        report["contact_sheet_path"] = str(sheet_path)

        click.echo("  -> running analysis stub (vision model not wired yet)")
        result = analysis_mod.analyze_contact_sheet(sheet_path, report)
        report.update({
            "visual_hook_summary": result.visual_hook_summary,
            "onscreen_text": result.onscreen_text,
            "emotional_mechanic": result.emotional_mechanic,
            "viewer_role": result.viewer_role,
            "emotions_triggered": result.emotions_triggered,
            "product_attachability_score": result.product_attachability_score,
            "transferability_score": result.transferability_score,
            "freshness_score": result.freshness_score,
            "cooked_score": result.cooked_score,
            "overall_opportunity_score": result.overall_opportunity_score,
            "hook_mutations": result.hook_mutations,
            "raw_analysis": result.raw_analysis,
        })
    except (VideoError, OSError) as e:
        report["error"] = f"{type(e).__name__}: {e}"
    finally:
        if not keep_temp:
            if video_path is not None:
                remove_video_file(video_path)
            remove_frame_dir(frames_dir)

    return report


def _run_pipeline(
    urls: list[str],
    paths,
    keep_temp: bool,
) -> list[str]:
    env = load_env()
    token = get_apify_token(env)

    click.echo(f"Submitting {len(urls)} URL(s) to Apify (clockworks/tiktok-video-scraper).")
    click.echo("Apify may charge per video. Cancel now (Ctrl+C) if this is unintended.")

    client = ApifyClient(token=token)
    try:
        items_raw, run_info = client.run_actor(urls)
    except ApifyError as e:
        raise click.ClickException(f"Apify run failed: {e}") from e

    if run_info.usage_total_usd is not None:
        click.echo(f"Apify run {run_info.run_id} succeeded. Cost: ${run_info.usage_total_usd:.4f}")
    else:
        click.echo(f"Apify run {run_info.run_id} succeeded.")

    items = normalize_items(items_raw, urls)
    paths.ensure()

    report_ids: list[str] = []
    for i, item in enumerate(items, start=1):
        click.echo(f"[{i}/{len(items)}] {item.submitted_url}")
        report = _process_one(
            item, paths, run_info, keep_temp=keep_temp, apify_token=token,
        )
        rid = insert_report(paths.db_path, report)
        report_ids.append(rid)
        if report.get("error"):
            click.echo(f"  ! error: {report['error']}")
        click.echo(f"  report id: {rid}")

    return report_ids


# ---- click group -----------------------------------------------------------

@click.group()
@click.option("--db", "db_path", type=click.Path(), default=None,
              help="Override SQLite DB path. Default: data/emotion_radar.db")
@click.option("--data-dir", "data_dir", type=click.Path(), default=None,
              help="Override data root. Default: data/")
@click.pass_context
def cli(ctx: click.Context, db_path: str | None, data_dir: str | None) -> None:
    ctx.ensure_object(dict)
    ctx.obj["paths"] = resolve_paths(data_dir=data_dir, db_path=db_path)


@cli.command("analyze-url")
@click.argument("url")
@click.option("--keep-temp", is_flag=True, default=False,
              help="Keep raw MP4 and frame JPEGs after the contact sheet is built.")
@click.pass_context
def analyze_url(ctx: click.Context, url: str, keep_temp: bool) -> None:
    """Analyze a single TikTok URL."""
    paths = ctx.obj["paths"]
    enforce_url_cap([url], confirm_large=True)  # single URL is always fine
    report_ids = _run_pipeline([url], paths, keep_temp=keep_temp)
    click.echo("\nDone.")
    for rid in report_ids:
        click.echo(f"  python -m emotion_radar show-report {rid}")


@cli.command("analyze-urls")
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--keep-temp", is_flag=True, default=False,
              help="Keep raw MP4 and frame JPEGs after the contact sheet is built.")
@click.option("--confirm-large", is_flag=True, default=False,
              help=f"Required if file contains more than {DEFAULT_MAX_URLS} URLs.")
@click.pass_context
def analyze_urls(ctx: click.Context, file_path: str, keep_temp: bool, confirm_large: bool) -> None:
    """Analyze multiple URLs (one per line; blanks and #-comments ignored)."""
    paths = ctx.obj["paths"]
    urls = _read_url_file(Path(file_path))
    try:
        enforce_url_cap(urls, confirm_large=confirm_large)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"Loaded {len(urls)} URL(s) from {file_path}.")
    report_ids = _run_pipeline(urls, paths, keep_temp=keep_temp)
    click.echo("\nDone.")
    for rid in report_ids:
        click.echo(f"  python -m emotion_radar show-report {rid}")


@cli.command("list-reports")
@click.option("--limit", default=50, show_default=True, type=int)
@click.pass_context
def list_reports_cmd(ctx: click.Context, limit: int) -> None:
    """List recent reports."""
    paths = ctx.obj["paths"]
    rows = list_reports(paths.db_path, limit=limit)
    if not rows:
        click.echo("(no reports yet)")
        return
    header = f"{'id':<14}{'platform':<10}{'creator':<22}{'views':>12}  {'score':>6}  {'created_at':<22}  caption"
    click.echo(header)
    click.echo("-" * len(header))
    for r in rows:
        metrics = r.get("metrics") or {}
        views = metrics.get("views")
        score = r.get("overall_opportunity_score")
        click.echo(
            f"{r['id']:<14}"
            f"{(r.get('platform') or ''):<10}"
            f"{_truncate(r.get('creator_username'), 20):<22}"
            f"{(views if views is not None else 0):>12}  "
            f"{(f'{score:.2f}' if isinstance(score, (int, float)) else '   -- '):>6}  "
            f"{(r.get('created_at') or ''):<22}  "
            f"{_truncate(r.get('caption'), 80)}"
        )


@cli.command("show-report")
@click.argument("report_id")
@click.pass_context
def show_report_cmd(ctx: click.Context, report_id: str) -> None:
    """Pretty-print a single report as JSON."""
    paths = ctx.obj["paths"]
    report = get_report(paths.db_path, report_id)
    if not report:
        raise click.ClickException(f"No report with id={report_id}")
    click.echo(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


def _resolve_contact_sheet(report: dict[str, Any]) -> Path:
    sheet_str = report.get("contact_sheet_path")
    if not sheet_str:
        raise click.ClickException(
            f"Report {report['id']} has no contact_sheet_path. "
            f"Re-run analyze-url or analyze-link to regenerate it."
        )
    sheet_path = Path(sheet_str)
    if not sheet_path.is_file():
        raise click.ClickException(
            f"Contact sheet missing on disk: {sheet_path}. "
            f"Re-run analyze-url or analyze-link to regenerate it."
        )
    return sheet_path


def _print_dry_run_prompts(
    report: dict[str, Any],
    sheet_path: Path,
    db_path: Path | None,
    three_pass: bool = True,
) -> None:
    """Print the prompts that would be sent. With three_pass=True
    (default), prints Pass 1, Pass 2, AND Pass 3 prompt templates.
    With three_pass=False, prints only Pass 1 and Pass 2."""
    pass1_user = analysis_mod.build_visual_event_user_prompt(report)
    pass1_placeholder = {"_placeholder": "Pass 1 evidence-layer JSON would be embedded here at runtime"}
    pass2_user = analysis_mod.build_hook_strategy_user_prompt(report, pass1_placeholder)

    click.echo("=== PASS 1 SYSTEM (Visual Event Extractor) ===")
    click.echo(analysis_mod.VISUAL_EVENT_SYSTEM_PROMPT)
    click.echo("\n=== PASS 1 USER ===")
    click.echo(pass1_user)
    click.echo("\n=== PASS 2 SYSTEM (Hook Strategist) ===")
    click.echo(analysis_mod.HOOK_STRATEGY_SYSTEM_PROMPT)
    click.echo("\n=== PASS 2 USER (template; Pass 1 JSON is embedded at runtime) ===")
    click.echo(pass2_user)

    if three_pass:
        pass2_placeholder = {
            "_placeholder": "Pass 2 variations + pioneer_concepts JSON would be embedded here at runtime"
        }
        taste = build_taste_summary(db_path) if db_path is not None else None
        pass3_user = analysis_mod.build_specificity_user_prompt(
            pass1_placeholder, pass2_placeholder, taste_profile=taste,
        )
        click.echo("\n=== PASS 3 SYSTEM (Specificity / Hook Scene Writer) ===")
        click.echo(analysis_mod.SPECIFICITY_SYSTEM_PROMPT)
        click.echo("\n=== PASS 3 USER (template; Pass 1 + Pass 2 JSON embedded at runtime) ===")
        click.echo(pass3_user)
        if taste:
            click.echo("\n(Pass 3 would be conditioned on the stored taste profile shown above.)")
        else:
            click.echo("\n(No stored taste feedback yet — Pass 3 would run without conditioning.)")

    click.echo(f"\n=== IMAGE (Pass 1 input) ===\n{sheet_path}")
    click.echo("\n(dry-run: no API call made)")


# Back-compat alias for any internal callers expecting the old name.
_print_two_pass_dry_run = _print_dry_run_prompts


def _build_pass_providers(env: dict[str, str]) -> tuple[VisionProvider, VisionProvider]:
    """Two providers — one per role. Models follow the
    VISION_EVENT_MODEL / HOOK_STRATEGY_MODEL / VISION_MODEL precedence
    encoded in providers.build_provider_for_role."""
    vision = build_provider_for_role(env, ROLE_VISION_EVENT)
    strategy = build_provider_for_role(env, ROLE_HOOK_STRATEGY)
    return vision, strategy


def _fields_from_result(result) -> dict[str, Any]:
    return {
        "visual_hook_summary": result.visual_hook_summary,
        "onscreen_text": result.onscreen_text,
        "emotional_mechanic": result.emotional_mechanic,
        "viewer_role": result.viewer_role,
        "emotions_triggered": result.emotions_triggered,
        "product_attachability_score": result.product_attachability_score,
        "transferability_score": result.transferability_score,
        "freshness_score": result.freshness_score,
        "cooked_score": result.cooked_score,
        "overall_opportunity_score": result.overall_opportunity_score,
        "hook_mutations": result.hook_mutations,
        "raw_analysis": result.raw_analysis,
    }


def _carry_source_metadata(previous_report: dict[str, Any], fields: dict[str, Any]) -> None:
    """The two-pass / three-pass merges build raw_analysis from scratch.
    Phase 7 adds raw_analysis.source_metadata for local seed clips; this
    helper carries that subfield across the overwrite so it survives
    vision passes."""
    prev_raw = previous_report.get("raw_analysis") or {}
    if not isinstance(prev_raw, dict):
        return
    prev_meta = prev_raw.get("source_metadata")
    if not isinstance(prev_meta, dict):
        return
    new_raw = fields.get("raw_analysis")
    if not isinstance(new_raw, dict):
        return
    new_raw["source_metadata"] = prev_meta


def _run_two_pass_and_update(
    report: dict[str, Any],
    sheet_path: Path,
    db_path: Path,
) -> dict[str, Any]:
    """Two-pass run (Pass 1 + Pass 2 only). Used when the user passes
    --no-specificity for debugging. The default analyze-link /
    analyze-report flow now uses three-pass."""
    env = load_env()
    try:
        vision_provider, strategy_provider = _build_pass_providers(env)
    except VisionProviderError as e:
        raise click.ClickException(str(e)) from e
    click.echo(
        f"Pass 1 (visual event)  : {vision_provider.model}\n"
        f"Pass 2 (hook strategy) : {strategy_provider.model}"
    )
    click.echo("(Vision API may incur cost.)")
    try:
        pass1, pass2 = analysis_mod.analyze_two_pass(
            sheet_path, report, vision_provider, strategy_provider,
        )
    except VisionProviderError as e:
        raise click.ClickException(f"Vision provider failed: {e}") from e
    except ValueError as e:
        raise click.ClickException(f"Could not parse model output: {e}") from e

    fields = _fields_from_result(analysis_mod.build_two_pass_analysis_result(pass1, pass2))
    _carry_source_metadata(report, fields)
    if not update_report_analysis(db_path, report["id"], fields):
        raise click.ClickException(
            f"DB update failed for report {report['id']} (row not found?)."
        )
    return fields


def _run_three_pass_and_update(
    report: dict[str, Any],
    sheet_path: Path,
    db_path: Path,
) -> dict[str, Any]:
    """Default flow: Pass 1 + Pass 2 + Pass 3 (specificity rewrite).
    Pass 3 is conditioned on the user's stored taste profile if any
    feedback rows exist; otherwise the profile is omitted."""
    env = load_env()
    try:
        vision_provider, strategy_provider = _build_pass_providers(env)
    except VisionProviderError as e:
        raise click.ClickException(str(e)) from e

    taste = build_taste_summary(db_path)
    click.echo(
        f"Pass 1 (visual event)  : {vision_provider.model}\n"
        f"Pass 2 (hook strategy) : {strategy_provider.model}\n"
        f"Pass 3 (specificity)   : {strategy_provider.model}"
    )
    if taste:
        click.echo("(Pass 3 will be conditioned on stored taste feedback.)")
    click.echo("(Vision API may incur cost.)")

    try:
        pass1, pass2, pass3 = analysis_mod.analyze_three_pass(
            sheet_path, report, vision_provider, strategy_provider,
            taste_profile=taste,
        )
    except VisionProviderError as e:
        raise click.ClickException(f"Vision provider failed: {e}") from e
    except ValueError as e:
        raise click.ClickException(f"Could not parse model output: {e}") from e

    fields = _fields_from_result(
        analysis_mod.build_three_pass_analysis_result(pass1, pass2, pass3)
    )
    _carry_source_metadata(report, fields)
    if not update_report_analysis(db_path, report["id"], fields):
        raise click.ClickException(
            f"DB update failed for report {report['id']} (row not found?)."
        )
    return fields


@cli.command("analyze-report")
@click.argument("report_id")
@click.option("--dry-run", is_flag=True, default=False,
              help="Print Pass 1 / 2 / 3 prompts, then exit. No API call.")
@click.option("--no-specificity", is_flag=True, default=False,
              help="Run only Pass 1 + Pass 2; skip the Phase-6 specificity pass.")
@click.pass_context
def analyze_report_cmd(
    ctx: click.Context, report_id: str, dry_run: bool, no_specificity: bool,
) -> None:
    """Re-run vision analysis on an existing report's contact sheet and
    write the structured hook intelligence back into the row. Defaults
    to three-pass (visual event -> hook strategy -> specificity)."""
    paths = ctx.obj["paths"]
    report = get_report(paths.db_path, report_id)
    if not report:
        raise click.ClickException(f"No report with id={report_id}")
    sheet_path = _resolve_contact_sheet(report)

    if dry_run:
        _print_dry_run_prompts(
            report, sheet_path, paths.db_path, three_pass=not no_specificity,
        )
        return

    if no_specificity:
        fields = _run_two_pass_and_update(report, sheet_path, paths.db_path)
        click.echo("\n=== Two-pass analysis ===")
    else:
        fields = _run_three_pass_and_update(report, sheet_path, paths.db_path)
        click.echo("\n=== Three-pass analysis ===")
    click.echo(json.dumps(fields, indent=2, ensure_ascii=False, sort_keys=True))
    click.echo(f"\nUpdated report {report_id}.")


def _fmt_score(v: Any) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) else "—"


_CREATIVE_DISTANCES = ("same_mechanic", "adjacent_leap", "big_swing", "wildcard")
_DISTANCE_LABELS = {
    "same_mechanic": "Same Mechanic",
    "adjacent_leap": "Adjacent Leap",
    "big_swing": "Big Swing",
    "wildcard": "Wildcard",
}


def _print_final_report(report: dict[str, Any]) -> None:
    """Phase-4 viral-mechanic summary for analyze-link.

    Prioritises: viral mechanic, scroll-stop reason, viewer role, comment
    trigger, share trigger, cooked elements, virality scores, and the 8
    broad hook concepts grouped by creative_distance. Product
    attachability is intentionally de-emphasised — the goal is the
    mechanic, not the product."""
    metrics = report.get("metrics") or {}
    raw = report.get("raw_analysis") or {}
    hsp: dict[str, Any] = {}
    if isinstance(raw, dict):
        v = raw.get("hook_strategy_pass")
        if isinstance(v, dict):
            hsp = v

    click.echo("\nEmotion Radar Report")
    click.echo("=" * 60)
    click.echo(f"Report ID:       {report.get('id')}")
    click.echo(f"Source:          {report.get('source_url') or report.get('submitted_url')}")
    creator = report.get("creator_username")
    nickname = report.get("creator_nickname")
    creator_line = f"@{creator}" + (f" ({nickname})" if nickname else "") if creator else "—"
    click.echo(f"Creator:         {creator_line}")
    click.echo(
        "Engagement:      "
        f"views={metrics.get('views')}  "
        f"likes={metrics.get('likes')}  "
        f"comments={metrics.get('comments')}  "
        f"shares={metrics.get('shares')}  "
        f"saves={metrics.get('saves')}"
    )
    click.echo(f"Contact Sheet:   {report.get('contact_sheet_path') or '—'}")
    if report.get("error"):
        click.echo(f"\nERROR: {report['error']}")
        return

    click.echo("")
    click.echo(f"Visual Hook:       {report.get('visual_hook_summary') or '—'}")
    click.echo(f"On-screen Text:    {report.get('onscreen_text') or '—'}")

    click.echo("\nViral Mechanic Analysis")
    click.echo("-" * 60)
    viral_mech = hsp.get("viral_mechanic") or report.get("emotional_mechanic")
    click.echo(f"Viral Mechanic:        {viral_mech or '—'}")
    click.echo(f"Why Stops Scroll:      {hsp.get('scroll_stop_reason') or '—'}")
    click.echo(f"Viewer Role:           {hsp.get('viewer_role') or report.get('viewer_role') or '—'}")
    click.echo(f"Comment Trigger:       {hsp.get('comment_trigger') or '—'}")
    click.echo(f"Share Trigger:         {hsp.get('share_trigger') or '—'}")
    click.echo(f"Emotional Pressure:    {hsp.get('emotional_pressure') or '—'}")
    click.echo(f"Freshness Angle:       {hsp.get('freshness_angle') or '—'}")
    cooked_elements = hsp.get("cooked_elements") or []
    if isinstance(cooked_elements, list) and cooked_elements:
        click.echo("Cooked Elements:")
        for item in cooked_elements:
            click.echo(f"  - {item}")
    else:
        click.echo("Cooked Elements:       —")
    emotions = report.get("emotions_triggered") or []
    if emotions:
        click.echo(f"Emotions Triggered:    {', '.join(emotions)}")

    click.echo("\nVirality Scores")
    click.echo("-" * 60)
    click.echo(f"  Scroll-stop strength:      {_fmt_score(hsp.get('scroll_stop_strength_score'))}")
    click.echo(f"  Comment likelihood:        {_fmt_score(hsp.get('comment_likelihood_score'))}")
    click.echo(f"  Share likelihood:          {_fmt_score(hsp.get('share_likelihood_score'))}")
    click.echo(f"  Viewer role strength:      {_fmt_score(hsp.get('viewer_role_strength_score'))}")
    click.echo(f"  Creative transfer pot.:    {_fmt_score(hsp.get('creative_transfer_potential_score'))}")
    click.echo(f"  Virality capability:       {_fmt_score(hsp.get('virality_capability_score'))}")
    click.echo(f"  Overall opportunity:       {_fmt_score(report.get('overall_opportunity_score'))}")
    click.echo(f"  Freshness (legacy):        {_fmt_score(report.get('freshness_score'))}")
    click.echo(f"  Cooked (legacy):           {_fmt_score(report.get('cooked_score'))}")

    concepts = report.get("hook_mutations") or []
    by_distance: dict[str, list[dict[str, Any]]] = {k: [] for k in _CREATIVE_DISTANCES}
    extras: list[dict[str, Any]] = []
    for m in concepts:
        if not isinstance(m, dict):
            continue
        dist = (m.get("creative_distance") or "").strip().lower()
        if dist in by_distance:
            by_distance[dist].append(m)
        else:
            extras.append(m)

    click.echo("\nBroad Hook Concepts")
    click.echo("-" * 60)
    for key in _CREATIVE_DISTANCES:
        bucket = by_distance[key]
        click.echo(f"\n  -- {_DISTANCE_LABELS[key]} ({len(bucket)}) --")
        if not bucket:
            click.echo("    (none)")
            continue
        for i, m in enumerate(bucket, start=1):
            click.echo(f"    {i}. {m.get('concept_name') or m.get('idea') or '(no name)'}")
            if m.get("first_2_seconds"):
                click.echo(f"       first 2 seconds:    {m['first_2_seconds']}")
            elif m.get("opening_scene"):
                click.echo(f"       first 2 seconds:    {m['opening_scene']}")
            if m.get("emotional_trigger"):
                click.echo(f"       emotional trigger:  {m['emotional_trigger']}")
            if m.get("viewer_role"):
                click.echo(f"       viewer role:        {m['viewer_role']}")
            if m.get("why_it_could_go_viral"):
                click.echo(f"       why viral:          {m['why_it_could_go_viral']}")
            elif m.get("why_it_might_work"):
                click.echo(f"       why viral:          {m['why_it_might_work']}")
            if m.get("what_to_avoid"):
                click.echo(f"       what to avoid:      {m['what_to_avoid']}")
            if m.get("believability_risk"):
                click.echo(f"       believability risk: {m['believability_risk']}")
            if m.get("cooked_risk"):
                click.echo(f"       cooked risk:        {m['cooked_risk']}")
            elif m.get("cringe_or_cooked_risk"):
                click.echo(f"       cooked risk:        {m['cringe_or_cooked_risk']}")

    if extras:
        click.echo(f"\n  -- Other ({len(extras)}, unrecognised creative_distance) --")
        for m in extras:
            t = m.get("type") or m.get("creative_distance") or "?"
            label = m.get("concept_name") or m.get("idea") or "(no name)"
            click.echo(f"    [{t}] {label}")

    # ----- Phase 5: Story Flow Match -----
    matched = hsp.get("matched_story_flows") if isinstance(hsp.get("matched_story_flows"), list) else []
    dominant = hsp.get("dominant_story_flow")
    observed_steps = hsp.get("story_flow_steps_observed") if isinstance(hsp.get("story_flow_steps_observed"), list) else []
    click.echo("\nStory Flow Match")
    click.echo("-" * 60)
    click.echo(f"  Dominant flow:        {dominant or '—'}")
    if matched:
        click.echo(f"  Matched flows ({len(matched)}):")
        for m in matched:
            if not isinstance(m, dict):
                continue
            conf = m.get("confidence")
            conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "—"
            click.echo(f"    - {m.get('id') or '?'} ({conf_str}): {m.get('why_matched') or ''}")
    else:
        click.echo("  Matched flows:        (none)")
    if observed_steps:
        click.echo("  Steps observed in source:")
        for step in observed_steps:
            click.echo(f"    * {step}")
    click.echo("\n  Phase 5 scores")
    click.echo(f"    Story-flow strength:        {_fmt_score(hsp.get('story_flow_strength_score'))}")
    click.echo(f"    Novelty beyond baseline:    {_fmt_score(hsp.get('novelty_beyond_baseline_score'))}")
    click.echo(f"    Ethical risk:               {_fmt_score(hsp.get('ethical_risk_score'))}")
    click.echo(f"    Cringe risk:                {_fmt_score(hsp.get('cringe_risk_score'))}")
    click.echo(f"    Breakout potential:         {_fmt_score(hsp.get('breakout_potential_score'))}")

    # ----- Phase 5: Variations -----
    variations = hsp.get("variations") if isinstance(hsp.get("variations"), list) else []
    click.echo(f"\nVariations ({len(variations)})")
    click.echo("-" * 60)
    if not variations:
        click.echo("  (none)")
    for i, v in enumerate(variations, start=1):
        if not isinstance(v, dict):
            continue
        click.echo(f"  {i}. {v.get('concept_name') or '(no name)'}  [flow: {v.get('story_flow_id') or '?'}]")
        if v.get("first_2_seconds"):
            click.echo(f"     first 2 seconds:        {v['first_2_seconds']}")
        if v.get("emotional_trigger"):
            click.echo(f"     emotional trigger:      {v['emotional_trigger']}")
        if v.get("viewer_role"):
            click.echo(f"     viewer role:            {v['viewer_role']}")
        if v.get("why_it_could_go_viral"):
            click.echo(f"     why viral:              {v['why_it_could_go_viral']}")
        if v.get("what_is_new"):
            click.echo(f"     what is new:            {v['what_is_new']}")
        if v.get("what_is_cooked_to_avoid"):
            click.echo(f"     cooked to avoid:        {v['what_is_cooked_to_avoid']}")
        if v.get("believability_risk"):
            click.echo(f"     believability risk:     {v['believability_risk']}")

    # ----- Phase 5: Pioneer Concepts (prominent — this is the user's goal) -----
    pioneers = hsp.get("pioneer_concepts") if isinstance(hsp.get("pioneer_concepts"), list) else []
    click.echo("\n" + ("=" * 60))
    click.echo(f"PIONEER CONCEPTS  ({len(pioneers)})")
    click.echo("=" * 60)
    if not pioneers:
        click.echo("  (none)")
    for i, p in enumerate(pioneers, start=1):
        if not isinstance(p, dict):
            continue
        click.echo(
            f"\n  [{i}] {p.get('concept_name') or '(no name)'}"
            f"   (inspired by: {p.get('inspired_by_story_flow_id') or '?'})"
        )
        if p.get("first_2_seconds"):
            click.echo(f"      first 2 seconds:           {p['first_2_seconds']}")
        if p.get("emotional_physics"):
            click.echo(f"      emotional physics:         {p['emotional_physics']}")
        if p.get("why_it_is_not_a_direct_copy"):
            click.echo(f"      not a direct copy:         {p['why_it_is_not_a_direct_copy']}")
        if p.get("why_it_could_be_breakout"):
            click.echo(f"      why it could be breakout:  {p['why_it_could_be_breakout']}")
        if p.get("viewer_comment_impulse"):
            click.echo(f"      viewer comment impulse:    {p['viewer_comment_impulse']}")
        if p.get("ethical_or_cringe_risk"):
            click.echo(f"      ethical / cringe risk:     {p['ethical_or_cringe_risk']}")

    # ----- Phase 6: Specific Hook Scenes (main actionable output) -----
    spec = {}
    if isinstance(raw, dict):
        v = raw.get("specificity_pass")
        if isinstance(v, dict):
            spec = v
    scenes = spec.get("scene_concepts") if isinstance(spec.get("scene_concepts"), list) else []
    click.echo("\n" + ("#" * 60))
    click.echo(f"### SPECIFIC HOOK SCENES  ({len(scenes)}) -- main actionable output")
    click.echo("#" * 60)
    weak_fixed = spec.get("weak_patterns_fixed") if isinstance(spec.get("weak_patterns_fixed"), list) else []
    if weak_fixed:
        click.echo("  Weak patterns the rewriter fixed:")
        for item in weak_fixed:
            click.echo(f"    - {item}")
    if spec.get("specificity_notes"):
        click.echo(f"  Notes: {spec['specificity_notes']}")
    if not scenes:
        click.echo("\n  (no scene concepts — run without --no-specificity for the Phase-6 rewrite)")
    for i, s in enumerate(scenes, start=1):
        if not isinstance(s, dict):
            continue
        click.echo(
            f"\n  [{i}] {s.get('specific_concept_name') or '(no name)'}"
            f"   (based on {s.get('source_type') or '?'}: "
            f"{s.get('source_concept_name') or '?'} | flow: {s.get('story_flow_id') or '?'})"
        )
        if s.get("first_2_seconds"):
            click.echo(f"      first 2 seconds:        {s['first_2_seconds']}")
        if s.get("onscreen_text"):
            click.echo(f"      onscreen text:          {s['onscreen_text']}")
        if s.get("visual_beat"):
            click.echo(f"      visual beat:            {s['visual_beat']}")
        if s.get("social_tension"):
            click.echo(f"      social tension:         {s['social_tension']}")
        if s.get("viewer_comment_impulse"):
            click.echo(f"      comment impulse:        {s['viewer_comment_impulse']}")
        if s.get("why_they_keep_watching"):
            click.echo(f"      why they keep watching: {s['why_they_keep_watching']}")
        if s.get("freshness_angle"):
            click.echo(f"      freshness angle:        {s['freshness_angle']}")
        if s.get("believability_risk"):
            click.echo(f"      believability risk:     {s['believability_risk']}")
        if s.get("cringe_risk"):
            click.echo(f"      cringe risk:            {s['cringe_risk']}")
        score = s.get("virality_potential_score")
        if score is not None:
            click.echo(f"      virality potential:     {_fmt_score(score)}")


def _print_evaluation(
    result: EvaluationResult,
    spec_path: Path | str,
    auto: bool,
    actual_mechanic: str | None,
) -> None:
    click.echo("\nEvaluation")
    label = " (auto, known video_id)" if auto else ""
    status = "PASS" if result.passed else "FAIL"
    click.echo(f"  status:    {status}{label}")
    click.echo(f"  spec:      {spec_path}")
    click.echo(
        f"  required:    {len(result.required_terms_matched)}/{result.required_terms_total} matched"
    )
    if result.required_terms_missing:
        click.echo("  missing required_terms:")
        for term in result.required_terms_missing:
            click.echo(f"    - {term}")
    if result.required_any_total or result.required_any_missing:
        click.echo(
            f"  required_any: "
            f"{len(result.required_any_matched)}/{result.required_any_total} groups matched"
        )
    for hit in result.required_any_matched:
        click.echo(f"    + matched \"{hit['matched']}\" from {hit['group']}")
    if result.required_any_missing:
        click.echo("  missing required_any groups (none of the synonyms appeared):")
        for group in result.required_any_missing:
            click.echo(f"    - any of: {group}")
    if result.forbidden_terms_present:
        click.echo("  forbidden terms PRESENT (taste regression):")
        for term in result.forbidden_terms_present:
            click.echo(f"    ! {term}")
    if result.expected_mechanic is not None or result.mechanic_any:
        verdict = "match" if result.mechanic_match else "MISMATCH"
        click.echo(f"  expected_mechanic: {verdict}")
        if result.expected_mechanic:
            click.echo(f"    expected:    {result.expected_mechanic}")
        if result.mechanic_any:
            click.echo(f"    or any of:   {result.mechanic_any}")
        click.echo(f"    actual:      {actual_mechanic or '—'}")
    if not result.passed:
        click.echo("\nCalibration failed. Do not trust this report yet.", err=True)


# ============================================================================
# Phase 7: local seed-clip ingestion (analyze-file, analyze-folder)
# ============================================================================

def _slugify_filename(stem: str) -> str:
    """Build a SQL/path-safe video_id from a filename stem. Non-alphanum
    runs collapse to single underscores; output is capped at 80 chars."""
    safe = "".join(ch if ch.isalnum() else "_" for ch in stem)
    # Collapse runs of underscores.
    while "__" in safe:
        safe = safe.replace("__", "_")
    safe = safe.strip("_") or "untitled"
    return safe[:80]


def _find_video_files(folder: Path, recursive: bool = False) -> list[Path]:
    """Find supported video files in `folder`. Non-recursive by default
    (matches the documented analyze-folder behavior). Results sorted by
    filename so per-run order is deterministic."""
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    matched: list[Path] = []
    for p in iterator:
        if not p.is_file():
            continue
        if p.suffix.lower() in SUPPORTED_VIDEO_EXTS:
            matched.append(p)
    matched.sort(key=lambda p: p.name)
    return matched


def _local_seed_report_stub(video_path: Path) -> dict[str, Any]:
    """Build the report dict for a local-seed-clip report row. No Apify
    fields. source_metadata lives under raw_analysis so subsequent
    vision passes can preserve it through the merge."""
    source_metadata = {
        "source_type": "drive_seed_clip",
        "source_filename": video_path.name,
        "known_viral": True,
        "analytics_available": False,
        "original_local_path": str(video_path),
    }
    return {
        "platform": "seed_clip",
        "source_url": None,
        "submitted_url": video_path.as_uri(),
        "video_id": _slugify_filename(video_path.stem),
        "creator_username": None,
        "creator_nickname": None,
        "caption": None,
        "metrics": {
            "views": None, "likes": None, "comments": None,
            "shares": None, "saves": None,
        },
        "duration": None,
        "cover_url": None,
        "video_download_url_saved": False,
        "apify_run_id": None,
        "apify_dataset_id": None,
        "apify_usage_usd": None,
        "apify_charged_events": None,
        "contact_sheet_path": None,
        "raw_analysis": {
            "analysis_mode": "stub",
            "source_metadata": source_metadata,
        },
        "error": None,
    }


def _ingest_local_video(
    paths,
    video_path: Path,
    keep_temp: bool = False,
) -> str:
    """Build the frames + contact sheet for a local video and insert a
    stub report row. Returns the new report_id. Errors during frame
    extraction land on the report's `error` field; the row is still
    inserted so the user can see what happened."""
    paths.ensure()
    report = _local_seed_report_stub(video_path)
    video_id = report["video_id"]
    frames_dir = paths.tmp_frames_dir / video_id

    try:
        click.echo(f"  extracting frames at {list(FRAME_TIMESTAMPS_SEC)}s ...")
        frame_paths = extract_frames(video_path, frames_dir, FRAME_TIMESTAMPS_SEC)
        sheet_path = paths.contact_sheets_dir / f"{video_id}.jpg"
        click.echo(f"  building contact sheet -> {sheet_path}")
        build_contact_sheet(
            frame_paths,
            list(FRAME_TIMESTAMPS_SEC),
            sheet_path,
        )
        report["contact_sheet_path"] = str(sheet_path)
    except (VideoError, OSError) as e:
        report["error"] = f"{type(e).__name__}: {e}"
    finally:
        if not keep_temp:
            remove_frame_dir(frames_dir)

    return insert_report(paths.db_path, report)


def _run_vision_phase_for_report(
    paths,
    report_id: str,
    *,
    no_vision: bool,
    no_specificity: bool,
    dry_run_vision: bool,
    skip_evaluation: bool,
    expected_path: str | None,
    auto_fixture_lookup: bool = True,
) -> None:
    """The shared "post-infrastructure" half used by analyze-link,
    analyze-file, and analyze-folder. Loads the report by id, decides
    no-vision / dry-run / two-pass / three-pass, prints the final
    summary, and runs calibration if a spec is available.

    `auto_fixture_lookup=False` lets analyze-folder suppress per-file
    fixture matching (a directory of seed clips shouldn't auto-evaluate
    each file against the Oliver canary)."""
    report = get_report(paths.db_path, report_id)
    if not report:
        raise click.ClickException(f"Could not load report {report_id}.")

    if report.get("error"):
        click.echo(f"\nPipeline error: {report['error']}")
        _print_final_report(report)
        return

    if no_vision:
        click.echo("\n--no-vision: skipping the two-pass analysis.")
        _print_final_report(report)
        return

    sheet_path = _resolve_contact_sheet(report)

    if dry_run_vision:
        _print_dry_run_prompts(
            report, sheet_path, paths.db_path, three_pass=not no_specificity,
        )
        _print_final_report(report)
        return

    if no_specificity:
        _run_two_pass_and_update(report, sheet_path, paths.db_path)
    else:
        _run_three_pass_and_update(report, sheet_path, paths.db_path)

    final_report = get_report(paths.db_path, report_id)
    if not final_report:
        raise click.ClickException("Report disappeared mid-pipeline.")

    _print_final_report(final_report)

    if skip_evaluation:
        click.echo("\n(--skip-evaluation: calibration check skipped)")
        return

    auto = False
    spec_path: Path | None = None
    if expected_path:
        spec_path = Path(expected_path)
    elif auto_fixture_lookup:
        video_id = (final_report.get("video_id") or "").strip()
        fixture = KNOWN_FIXTURES.get(video_id)
        if fixture and fixture.is_file():
            spec_path = fixture
            auto = True

    if not spec_path:
        return

    try:
        spec = load_expected(spec_path)
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"\nCould not run evaluation: {e}", err=True)
        return

    eval_result = evaluate_report_fn(final_report, spec)
    _print_evaluation(
        eval_result,
        spec_path=spec_path,
        auto=auto,
        actual_mechanic=final_report.get("emotional_mechanic"),
    )


def _print_batch_summary(
    db_path: Path,
    report_ids: list[str],
    failures: list[tuple[str, str]],
) -> None:
    click.echo("\n" + ("=" * 60))
    click.echo("BATCH SUMMARY")
    click.echo("=" * 60)
    click.echo(f"  Analyzed: {len(report_ids)}")
    click.echo(f"  Failed:   {len(failures)}")
    if report_ids:
        click.echo("\n  Report IDs:")
        for rid in report_ids:
            click.echo(f"    - {rid}")
        flow_counts: dict[str, int] = {}
        mechanics: list[tuple[str, str]] = []
        for rid in report_ids:
            row = get_report(db_path, rid)
            if not row:
                continue
            raw = row.get("raw_analysis") or {}
            hsp = raw.get("hook_strategy_pass") if isinstance(raw, dict) else None
            if isinstance(hsp, dict):
                dom = (hsp.get("dominant_story_flow") or "").strip()
                if dom:
                    flow_counts[dom] = flow_counts.get(dom, 0) + 1
                mech = hsp.get("viral_mechanic")
                if isinstance(mech, str) and mech.strip():
                    mechanics.append((rid, mech))
        if flow_counts:
            click.echo("\n  Dominant story flows:")
            for flow, n in sorted(flow_counts.items(), key=lambda kv: -kv[1]):
                click.echo(f"    {flow}: {n}")
        if mechanics:
            click.echo("\n  Viral mechanics:")
            for rid, mech in mechanics:
                click.echo(f"    [{rid}] {mech}")
    if failures:
        click.echo("\n  Failures:")
        for path, err in failures:
            click.echo(f"    - {path}: {err}")


@cli.command("analyze-file")
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("--keep-temp", is_flag=True, default=False,
              help="Keep extracted frames for debugging.")
@click.option("--dry-run-vision", is_flag=True, default=False,
              help="Run frame extraction + contact sheet, then print all vision prompts; no API call.")
@click.option("--no-vision", is_flag=True, default=False,
              help="Skip the vision passes entirely; produce only the infrastructure report stub.")
@click.option("--no-specificity", is_flag=True, default=False,
              help="Run only Pass 1 + Pass 2; skip the Phase-6 specificity pass.")
@click.option("--skip-evaluation", is_flag=True, default=False,
              help="Skip the calibration check, even for known video ids.")
@click.option("--expected", "expected_path", type=click.Path(exists=True, dir_okay=False),
              default=None, help="Calibration spec path.")
@click.pass_context
def analyze_file_cmd(
    ctx: click.Context,
    path: str,
    keep_temp: bool,
    dry_run_vision: bool,
    no_vision: bool,
    no_specificity: bool,
    skip_evaluation: bool,
    expected_path: str | None,
) -> None:
    """Analyze ONE local video file with the three-pass pipeline. No
    Apify. Used for Drive seed clips and other local sources."""
    paths = ctx.obj["paths"]
    video_path = Path(path)
    if video_path.suffix.lower() not in SUPPORTED_VIDEO_EXTS:
        raise click.ClickException(
            f"Unsupported video extension: {video_path.suffix!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_VIDEO_EXTS))}"
        )

    click.echo(f"Analyzing local file: {video_path}")
    click.echo("(seed clip — no Apify call, no analytics available.)")
    report_id = _ingest_local_video(paths, video_path, keep_temp=keep_temp)
    click.echo(f"  report id: {report_id}")

    _run_vision_phase_for_report(
        paths, report_id,
        no_vision=no_vision,
        no_specificity=no_specificity,
        dry_run_vision=dry_run_vision,
        skip_evaluation=skip_evaluation,
        expected_path=expected_path,
        auto_fixture_lookup=True,
    )


@cli.command("analyze-folder")
@click.argument("folder", type=click.Path(exists=True, file_okay=False))
@click.option("--limit", default=DEFAULT_FOLDER_LIMIT, show_default=True, type=int,
              help=f"Max files to analyze. Default {DEFAULT_FOLDER_LIMIT}; raising it prints a cost warning.")
@click.option("--recursive", is_flag=True, default=False,
              help="Walk subdirectories. Default: only the immediate folder.")
@click.option("--keep-temp", is_flag=True, default=False,
              help="Keep extracted frames for debugging.")
@click.option("--no-vision", is_flag=True, default=False,
              help="Skip the vision passes entirely; produce only stubs.")
@click.option("--no-specificity", is_flag=True, default=False,
              help="Run only Pass 1 + Pass 2; skip the Phase-6 specificity pass.")
@click.pass_context
def analyze_folder_cmd(
    ctx: click.Context,
    folder: str,
    limit: int,
    recursive: bool,
    keep_temp: bool,
    no_vision: bool,
    no_specificity: bool,
) -> None:
    """Analyze multiple local video files (Drive seed clips) in a folder.
    Sorted by filename for deterministic order. Continues on per-file
    failure and prints a batch summary at the end. Does NOT auto-run
    calibration per file (seed clips have no analytics)."""
    paths = ctx.obj["paths"]
    folder_path = Path(folder)
    files = _find_video_files(folder_path, recursive=recursive)
    if not files:
        raise click.ClickException(
            f"No supported video files found in {folder_path}. "
            f"Looking for: {', '.join(sorted(SUPPORTED_VIDEO_EXTS))}"
        )

    if limit > DEFAULT_FOLDER_LIMIT:
        click.echo(
            f"WARNING: --limit {limit} exceeds the safe default ({DEFAULT_FOLDER_LIMIT}). "
            f"Each file runs the full three-pass vision pipeline; vision API "
            f"calls may incur cost. Press Ctrl+C now if this is unintended."
        )

    selected = files[:limit]
    click.echo(
        f"Found {len(files)} video file(s) in {folder_path}; "
        f"analyzing {len(selected)} (sorted by filename)."
    )

    report_ids: list[str] = []
    failures: list[tuple[str, str]] = []
    for i, video_path in enumerate(selected, start=1):
        click.echo(f"\n[{i}/{len(selected)}] {video_path.name}")
        try:
            report_id = _ingest_local_video(paths, video_path, keep_temp=keep_temp)
            click.echo(f"  report id: {report_id}")
            _run_vision_phase_for_report(
                paths, report_id,
                no_vision=no_vision,
                no_specificity=no_specificity,
                dry_run_vision=False,
                # batch mode never auto-evaluates per file; instead of
                # passing skip_evaluation=True (which would print a
                # per-file notice), we suppress auto-lookup and let
                # spec_path stay None → silent return.
                skip_evaluation=False,
                expected_path=None,
                auto_fixture_lookup=False,
            )
            report_ids.append(report_id)
        except click.ClickException as e:
            failures.append((str(video_path), e.message))
            click.echo(f"  ! failed: {e.message}", err=True)
        except Exception as e:  # noqa: BLE001 - one bad file should not kill the batch
            failures.append((str(video_path), f"{type(e).__name__}: {e}"))
            click.echo(f"  ! failed: {type(e).__name__}: {e}", err=True)

    _print_batch_summary(paths.db_path, report_ids, failures)


@cli.command("analyze-link")
@click.argument("url")
@click.option("--keep-temp", is_flag=True, default=False,
              help="Keep raw MP4 and frame JPEGs after the contact sheet is built.")
@click.option("--dry-run-vision", is_flag=True, default=False,
              help="Run Apify/video/contact sheet, then print all vision prompts; no API call.")
@click.option("--no-vision", is_flag=True, default=False,
              help="Skip the vision passes entirely; produce only the infrastructure report stub.")
@click.option("--no-specificity", is_flag=True, default=False,
              help="Run only Pass 1 + Pass 2; skip the Phase-6 specificity pass.")
@click.option("--skip-evaluation", is_flag=True, default=False,
              help="Skip the calibration check, even for known video ids.")
@click.option("--expected", "expected_path", type=click.Path(exists=True, dir_okay=False),
              default=None, help="Calibration spec path (overrides any auto-fixture).")
@click.pass_context
def analyze_link_cmd(
    ctx: click.Context,
    url: str,
    keep_temp: bool,
    dry_run_vision: bool,
    no_vision: bool,
    no_specificity: bool,
    skip_evaluation: bool,
    expected_path: str | None,
) -> None:
    """One-shot: Apify -> contact sheet -> two-pass vision -> report.

    The user supplies a single TikTok URL; everything else is automatic.
    --dry-run-vision still runs Apify/video/contact sheet but skips the
    vision API. --no-vision stops before the vision step. If the
    analyzed video_id matches a known calibration fixture (e.g. the
    Oliver HTTYD-lamp video), the spec is auto-evaluated unless
    --skip-evaluation is passed."""
    paths = ctx.obj["paths"]
    enforce_url_cap([url], confirm_large=True)

    # --- 1. Apify + video + contact sheet (report stub inserted in DB) ---
    report_ids = _run_pipeline([url], paths, keep_temp=keep_temp)
    if not report_ids:
        raise click.ClickException("Pipeline produced no report.")
    report_id = report_ids[0]
    report = get_report(paths.db_path, report_id)
    if not report:
        raise click.ClickException(f"Could not load freshly-inserted report {report_id}.")

    # --- 2-6. Vision passes + summary + calibration (shared helper) ---
    _run_vision_phase_for_report(
        paths, report_id,
        no_vision=no_vision,
        no_specificity=no_specificity,
        dry_run_vision=dry_run_vision,
        skip_evaluation=skip_evaluation,
        expected_path=expected_path,
        auto_fixture_lookup=True,
    )


@cli.command("evaluate-report")
@click.argument("report_id")
@click.option("--expected", "expected_path", type=click.Path(exists=True, dir_okay=False),
              required=True, help="Path to expected.json calibration spec.")
@click.pass_context
def evaluate_report_cmd(ctx: click.Context, report_id: str, expected_path: str) -> None:
    """Compare an analyzed report against an expected.json spec
    (required_terms / required_any groups / forbidden_terms /
    expected_mechanic / mechanic_any). Simple case-insensitive
    substring check; a fast canary, not a semantic judge. Exit code
    is non-zero on failure."""
    paths = ctx.obj["paths"]
    report = get_report(paths.db_path, report_id)
    if not report:
        raise click.ClickException(f"No report with id={report_id}")
    try:
        expected = load_expected(expected_path)
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e)) from e

    result = evaluate_report_fn(report, expected)

    status = "PASS" if result.passed else "FAIL"
    click.echo(f"=== Evaluation: {status} ===")
    click.echo(f"report:   {report_id}")
    click.echo(f"expected: {expected_path}")
    click.echo(
        f"required_terms: {len(result.required_terms_matched)}/"
        f"{result.required_terms_total} matched"
    )
    if result.required_terms_matched:
        for term in result.required_terms_matched:
            click.echo(f"  + {term}")
    if result.required_terms_missing:
        click.echo("missing required_terms:")
        for term in result.required_terms_missing:
            click.echo(f"  - {term}")
    if result.required_any_total or result.required_any_missing:
        click.echo(
            f"required_any: {len(result.required_any_matched)}/"
            f"{result.required_any_total} groups matched"
        )
    for hit in result.required_any_matched:
        click.echo(f"  + matched \"{hit['matched']}\" from {hit['group']}")
    if result.required_any_missing:
        click.echo("missing required_any groups (none of the synonyms appeared):")
        for group in result.required_any_missing:
            click.echo(f"  - any of: {group}")
    if result.forbidden_terms_present:
        click.echo("forbidden terms PRESENT (taste regression):")
        for term in result.forbidden_terms_present:
            click.echo(f"  ! {term}")
    if result.expected_mechanic is not None or result.mechanic_any:
        verdict = "match" if result.mechanic_match else "MISMATCH"
        click.echo(f"expected_mechanic: {verdict}")
        if result.expected_mechanic:
            click.echo(f"  expected:  {result.expected_mechanic}")
        if result.mechanic_any:
            click.echo(f"  or any of: {result.mechanic_any}")
        click.echo(f"  actual:    {report.get('emotional_mechanic')}")

    if not result.passed:
        # Make CI / shell-pipeline failures obvious.
        ctx.exit(1)


def _get_scene_concepts(report: dict[str, Any]) -> list[dict[str, Any]]:
    raw = report.get("raw_analysis") or {}
    if not isinstance(raw, dict):
        return []
    spec = raw.get("specificity_pass") or {}
    if not isinstance(spec, dict):
        return []
    scenes = spec.get("scene_concepts")
    return scenes if isinstance(scenes, list) else []


@cli.command("list-scenes")
@click.argument("report_id")
@click.pass_context
def list_scenes_cmd(ctx: click.Context, report_id: str) -> None:
    """List Phase-6 scene concepts (with indexes) for a report, suitable
    for `rate-scene REPORT_ID INDEX`."""
    paths = ctx.obj["paths"]
    report = get_report(paths.db_path, report_id)
    if not report:
        raise click.ClickException(f"No report with id={report_id}")
    scenes = _get_scene_concepts(report)
    if not scenes:
        click.echo(
            "(no scene_concepts in this report — re-run analyze-link or "
            "analyze-report without --no-specificity to populate them)"
        )
        return
    for i, s in enumerate(scenes, start=1):
        if not isinstance(s, dict):
            continue
        name = s.get("specific_concept_name") or s.get("source_concept_name") or "(no name)"
        click.echo(f"[{i}] {name}")
        if s.get("first_2_seconds"):
            click.echo(f"    first 2 seconds: {s['first_2_seconds']}")
        if s.get("onscreen_text"):
            click.echo(f"    onscreen text:   {s['onscreen_text']}")
        score = s.get("virality_potential_score")
        if score is not None:
            click.echo(f"    virality score:  {_fmt_score(score)}")


@cli.command("rate-scene")
@click.argument("report_id")
@click.argument("index", type=int)
@click.option("--rating", required=True,
              type=click.Choice(list(ALLOWED_RATINGS), case_sensitive=False),
              help="One of: " + ", ".join(ALLOWED_RATINGS))
@click.option("--note", default=None, help="Optional free-text note.")
@click.pass_context
def rate_scene_cmd(
    ctx: click.Context, report_id: str, index: int, rating: str, note: str | None,
) -> None:
    """Record a rating for a scene concept by 1-based index. Used to
    build the taste profile that conditions future Pass 3 runs."""
    paths = ctx.obj["paths"]
    report = get_report(paths.db_path, report_id)
    if not report:
        raise click.ClickException(f"No report with id={report_id}")
    scenes = _get_scene_concepts(report)
    if not scenes:
        raise click.ClickException(
            "Report has no scene_concepts — re-run analyze-report "
            "without --no-specificity first."
        )
    if index < 1 or index > len(scenes):
        raise click.ClickException(
            f"Scene index {index} out of range (1..{len(scenes)})."
        )
    scene = scenes[index - 1]
    if not isinstance(scene, dict):
        raise click.ClickException(f"Scene at index {index} is malformed.")

    source_type = scene.get("source_type") or "scene_concept"
    concept_name = (
        scene.get("specific_concept_name")
        or scene.get("source_concept_name")
        or "(unnamed)"
    )
    try:
        feedback_id = insert_feedback(
            paths.db_path,
            report_id=report_id,
            concept_source_type=source_type,
            concept_name=concept_name,
            concept_index=index,
            rating=rating.lower(),
            note=note,
        )
    except FeedbackError as e:
        raise click.ClickException(str(e)) from e

    click.echo(
        f"Recorded feedback #{feedback_id}: scene[{index}] '{concept_name}' = {rating.lower()}"
        + (f"  -- {note}" if note else "")
    )


@cli.command("list-feedback")
@click.option("--limit", default=20, show_default=True, type=int,
              help="Newest N feedback rows.")
@click.option("--report-id", "report_id", default=None,
              help="Filter to one report.")
@click.pass_context
def list_feedback_cmd(
    ctx: click.Context, limit: int, report_id: str | None,
) -> None:
    """List recent scene-concept feedback (newest first)."""
    paths = ctx.obj["paths"]
    rows = (
        get_report_feedback(paths.db_path, report_id)
        if report_id else list_feedback(paths.db_path, limit=limit)
    )
    if not rows:
        click.echo("(no feedback yet)")
        return
    for r in rows[:limit]:
        note_part = f"  -- {r['note']}" if r.get("note") else ""
        click.echo(
            f"#{r['id']:<4} [{r['rating']:<6}] "
            f"{r['concept_name']}  "
            f"(report={r['report_id']}, scene={r['concept_index']}, "
            f"source={r['concept_source_type']}){note_part}"
        )


@cli.command("taste-summary")
@click.option("--limit", default=50, show_default=True, type=int,
              help="Build summary from the newest N feedback rows.")
@click.pass_context
def taste_summary_cmd(ctx: click.Context, limit: int) -> None:
    """Print the compact taste profile that Pass 3 will use to condition
    future scene rewrites."""
    paths = ctx.obj["paths"]
    summary = build_taste_summary(paths.db_path, limit=limit)
    if not summary:
        click.echo(
            "(no feedback yet — rate scenes with `rate-scene REPORT_ID INDEX "
            "--rating fire` to build a taste profile)"
        )
        return
    click.echo("=== Stored taste profile (used to condition Pass 3) ===")
    click.echo(summary)


@cli.command("cleanup-temp")
@click.pass_context
def cleanup_temp_cmd(ctx: click.Context) -> None:
    """Delete temp videos and frames. Contact sheets and DB are preserved."""
    paths = ctx.obj["paths"]
    summary = cleanup_temp_dirs(paths.tmp_videos_dir, paths.tmp_frames_dir)
    click.echo(
        f"Removed {summary.videos_removed} temp video entries "
        f"and {summary.frame_dirs_removed} frame dirs."
    )


def main(argv: list[str] | None = None) -> int:
    try:
        cli.main(args=argv, prog_name="emotion_radar", standalone_mode=False)
    except click.ClickException as e:
        e.show(file=sys.stderr)
        return e.exit_code
    except click.exceptions.Abort:
        click.echo("Aborted.", err=True)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
