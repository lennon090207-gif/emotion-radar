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
from .db import get_report, insert_report, list_reports, update_report_analysis
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


def _print_two_pass_dry_run(report: dict[str, Any], sheet_path: Path) -> None:
    """Print both Pass 1 and Pass 2 prompts. Pass 2's user prompt is
    built with a placeholder evidence layer since the real Pass 1
    output doesn't exist yet."""
    pass1_user = analysis_mod.build_visual_event_user_prompt(report)
    placeholder = {"_placeholder": "Pass 1 evidence-layer JSON would be embedded here"}
    pass2_user = analysis_mod.build_hook_strategy_user_prompt(report, placeholder)
    click.echo("=== PASS 1 SYSTEM (Visual Event Extractor) ===")
    click.echo(analysis_mod.VISUAL_EVENT_SYSTEM_PROMPT)
    click.echo("\n=== PASS 1 USER ===")
    click.echo(pass1_user)
    click.echo("\n=== PASS 2 SYSTEM (Hook Strategist) ===")
    click.echo(analysis_mod.HOOK_STRATEGY_SYSTEM_PROMPT)
    click.echo("\n=== PASS 2 USER (template; Pass 1 JSON is embedded at runtime) ===")
    click.echo(pass2_user)
    click.echo(f"\n=== IMAGE (Pass 1 input) ===\n{sheet_path}")
    click.echo("\n(dry-run: no API call made)")


def _build_pass_providers(env: dict[str, str]) -> tuple[VisionProvider, VisionProvider]:
    """Two providers — one per role. Models follow the
    VISION_EVENT_MODEL / HOOK_STRATEGY_MODEL / VISION_MODEL precedence
    encoded in providers.build_provider_for_role."""
    vision = build_provider_for_role(env, ROLE_VISION_EVENT)
    strategy = build_provider_for_role(env, ROLE_HOOK_STRATEGY)
    return vision, strategy


def _run_two_pass_and_update(
    report: dict[str, Any],
    sheet_path: Path,
    db_path: Path,
) -> dict[str, Any]:
    """Real two-pass run. Builds providers from env, runs Pass 1 and
    Pass 2, merges into AnalysisResult, writes back to the row,
    returns the merged-fields dict for printing."""
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

    result = analysis_mod.build_two_pass_analysis_result(pass1, pass2)
    fields = {
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
    if not update_report_analysis(db_path, report["id"], fields):
        raise click.ClickException(
            f"DB update failed for report {report['id']} (row not found?)."
        )
    return fields


@cli.command("analyze-report")
@click.argument("report_id")
@click.option("--dry-run", is_flag=True, default=False,
              help="Print both Pass 1 and Pass 2 prompts, then exit. No API call.")
@click.pass_context
def analyze_report_cmd(ctx: click.Context, report_id: str, dry_run: bool) -> None:
    """Re-run two-pass vision analysis on an existing report's contact
    sheet and write the structured hook intelligence back into the row."""
    paths = ctx.obj["paths"]
    report = get_report(paths.db_path, report_id)
    if not report:
        raise click.ClickException(f"No report with id={report_id}")
    sheet_path = _resolve_contact_sheet(report)

    if dry_run:
        _print_two_pass_dry_run(report, sheet_path)
        return

    fields = _run_two_pass_and_update(report, sheet_path, paths.db_path)
    click.echo("\n=== Two-pass analysis ===")
    click.echo(json.dumps(fields, indent=2, ensure_ascii=False, sort_keys=True))
    click.echo(f"\nUpdated report {report_id}.")


def _fmt_score(v: Any) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) else "—"


def _print_final_report(report: dict[str, Any]) -> None:
    """Concise markdown-ish summary for analyze-link."""
    metrics = report.get("metrics") or {}
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
    click.echo(f"Visual Hook:        {report.get('visual_hook_summary') or '—'}")
    click.echo(f"On-screen Text:     {report.get('onscreen_text') or '—'}")
    click.echo(f"Emotional Mechanic: {report.get('emotional_mechanic') or '—'}")
    click.echo(f"Viewer Role:        {report.get('viewer_role') or '—'}")
    emotions = report.get("emotions_triggered") or []
    if emotions:
        click.echo(f"Emotions:           {', '.join(emotions)}")

    click.echo("\nScores")
    click.echo(f"  Freshness:            {_fmt_score(report.get('freshness_score'))}")
    click.echo(f"  Cooked:               {_fmt_score(report.get('cooked_score'))}")
    click.echo(f"  Transferability:      {_fmt_score(report.get('transferability_score'))}")
    click.echo(f"  Product Attachability:{_fmt_score(report.get('product_attachability_score'))}")
    click.echo(f"  Overall Opportunity:  {_fmt_score(report.get('overall_opportunity_score'))}")

    mutations = report.get("hook_mutations") or []
    by_type: dict[str, list[dict[str, Any]]] = {"safe": [], "fresh": [], "big_swing": []}
    extras: list[dict[str, Any]] = []
    for m in mutations:
        if not isinstance(m, dict):
            continue
        t = (m.get("type") or "").strip().lower()
        if t in by_type:
            by_type[t].append(m)
        else:
            extras.append(m)

    click.echo("\nHook Ideas")
    for label, key in (("Safe", "safe"), ("Fresh", "fresh"), ("Big Swing", "big_swing")):
        bucket = by_type[key]
        click.echo(f"\n  -- {label} ({len(bucket)}) --")
        if not bucket:
            click.echo("    (none)")
            continue
        for i, m in enumerate(bucket, start=1):
            click.echo(f"    {i}. {m.get('idea') or '(no idea)'}")
            if m.get("opening_scene"):
                click.echo(f"       opening_scene:   {m['opening_scene']}")
            if m.get("onscreen_text"):
                click.echo(f"       onscreen_text:   {m['onscreen_text']}")
            if m.get("product_niche_fit"):
                click.echo(f"       product/niche:   {m['product_niche_fit']}")
            if m.get("why_it_might_work"):
                click.echo(f"       why_it_works:    {m['why_it_might_work']}")
            if m.get("cringe_or_cooked_risk"):
                click.echo(f"       risk:            {m['cringe_or_cooked_risk']}")
            if m.get("production_difficulty"):
                click.echo(f"       difficulty:      {m['production_difficulty']}")
    if extras:
        click.echo(f"\n  -- Other ({len(extras)}, unrecognised type) --")
        for m in extras:
            click.echo(f"    - {m.get('idea') or m}")


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
        f"  required:  {len(result.required_terms_matched)}/{result.required_terms_total} matched"
    )
    if result.required_terms_missing:
        click.echo("  missing terms:")
        for term in result.required_terms_missing:
            click.echo(f"    - {term}")
    if result.forbidden_terms_present:
        click.echo("  forbidden terms PRESENT (taste regression):")
        for term in result.forbidden_terms_present:
            click.echo(f"    ! {term}")
    if result.expected_mechanic is not None:
        verdict = "match" if result.mechanic_match else "MISMATCH"
        click.echo(f"  expected_mechanic: {verdict}")
        click.echo(f"    expected: {result.expected_mechanic}")
        click.echo(f"    actual:   {actual_mechanic or '—'}")
    if not result.passed:
        click.echo("\nCalibration failed. Do not trust this report yet.", err=True)


@cli.command("analyze-link")
@click.argument("url")
@click.option("--keep-temp", is_flag=True, default=False,
              help="Keep raw MP4 and frame JPEGs after the contact sheet is built.")
@click.option("--dry-run-vision", is_flag=True, default=False,
              help="Run Apify/video/contact sheet, then print both vision prompts; no API call.")
@click.option("--no-vision", is_flag=True, default=False,
              help="Skip the vision passes entirely; produce only the infrastructure report stub.")
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

    # --- 2. If the pipeline errored, surface and exit. ---
    if report.get("error"):
        click.echo(f"\nPipeline error: {report['error']}")
        _print_final_report(report)
        return

    # --- 3. Decide on vision behavior. ---
    if no_vision:
        click.echo("\n--no-vision: skipping the two-pass analysis.")
        _print_final_report(report)
        return

    sheet_path = _resolve_contact_sheet(report)

    if dry_run_vision:
        _print_two_pass_dry_run(report, sheet_path)
        _print_final_report(report)
        return

    # --- 4. Two-pass vision + DB update. ---
    _run_two_pass_and_update(report, sheet_path, paths.db_path)
    final_report = get_report(paths.db_path, report_id)
    if not final_report:
        raise click.ClickException("Report disappeared mid-pipeline.")

    # --- 5. Final summary. ---
    _print_final_report(final_report)

    # --- 6. Calibration. Manual --expected wins; otherwise auto on known ids. ---
    if skip_evaluation:
        click.echo("\n(--skip-evaluation: calibration check skipped)")
        return

    auto = False
    spec_path: Path | None = None
    if expected_path:
        spec_path = Path(expected_path)
    else:
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


@cli.command("evaluate-report")
@click.argument("report_id")
@click.option("--expected", "expected_path", type=click.Path(exists=True, dir_okay=False),
              required=True, help="Path to expected.json calibration spec.")
@click.pass_context
def evaluate_report_cmd(ctx: click.Context, report_id: str, expected_path: str) -> None:
    """Compare an analyzed report against an expected.json spec
    (required_terms / forbidden_terms / expected_mechanic). Simple
    case-insensitive substring check — a fast canary, not a semantic
    judge. Exit code is non-zero on failure."""
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
        click.echo("missing terms:")
        for term in result.required_terms_missing:
            click.echo(f"  - {term}")
    if result.forbidden_terms_present:
        click.echo("forbidden terms PRESENT (taste regression):")
        for term in result.forbidden_terms_present:
            click.echo(f"  ! {term}")
    if result.expected_mechanic is not None:
        verdict = "match" if result.mechanic_match else "MISMATCH"
        click.echo(f"expected_mechanic: {verdict}")
        click.echo(f"  expected: {result.expected_mechanic}")
        click.echo(f"  actual:   {report.get('emotional_mechanic')}")

    if not result.passed:
        # Make CI / shell-pipeline failures obvious.
        ctx.exit(1)


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
