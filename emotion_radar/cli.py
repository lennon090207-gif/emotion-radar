"""CLI entrypoints.

Commands:
  analyze-url URL                  — analyze one TikTok URL
  analyze-urls FILE                — analyze URLs listed in FILE (one per line)
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
from .db import insert_report, get_report, list_reports
from .models import NormalizedItem
from .video import build_contact_sheet, download_video, extract_frames, VideoError


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
) -> dict[str, Any]:
    """Process a single normalized item through download → frames →
    contact sheet → analysis stub → report dict. Errors in any step are
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

    video_id = item.video_id or "unknown"
    video_path: Path | None = None
    frames_dir = paths.tmp_frames_dir / video_id
    try:
        click.echo(f"  → downloading video for {video_id} ...")
        video_path = download_video(
            item.video_download_url,
            paths.tmp_videos_dir,
            video_id,
        )
        click.echo(f"  → extracting frames at {list(FRAME_TIMESTAMPS_SEC)}s ...")
        frame_paths = extract_frames(video_path, frames_dir, FRAME_TIMESTAMPS_SEC)
        sheet_path = paths.contact_sheets_dir / f"{video_id}.jpg"
        click.echo(f"  → building contact sheet → {sheet_path}")
        build_contact_sheet(
            frame_paths,
            list(FRAME_TIMESTAMPS_SEC),
            sheet_path,
        )
        report["contact_sheet_path"] = str(sheet_path)

        click.echo("  → running analysis stub (vision model not wired yet)")
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
        report = _process_one(item, paths, run_info, keep_temp=keep_temp)
        rid = insert_report(paths.db_path, report)
        report_ids.append(rid)
        if report.get("error"):
            click.echo(f"  ! error: {report['error']}")
        click.echo(f"  ✓ report id: {rid}")

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
