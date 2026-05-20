"""SQLite persistence for reports.

One table, `reports`, holds both raw metadata and analysis fields. Analysis
fields are nullable so this MVP can write rows before the vision model is
wired up.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id                            TEXT PRIMARY KEY,
    created_at                    TEXT NOT NULL,
    platform                      TEXT NOT NULL,
    source_url                    TEXT,
    submitted_url                 TEXT NOT NULL,
    video_id                      TEXT,
    creator_username              TEXT,
    creator_nickname              TEXT,
    caption                       TEXT,
    metrics_json                  TEXT,
    duration                      REAL,
    contact_sheet_path            TEXT,
    cover_url                     TEXT,
    video_download_url_saved      INTEGER NOT NULL DEFAULT 0,
    apify_run_id                  TEXT,
    apify_dataset_id              TEXT,
    apify_usage_usd               REAL,
    apify_charged_events_json     TEXT,
    visual_hook_summary           TEXT,
    onscreen_text                 TEXT,
    emotional_mechanic            TEXT,
    viewer_role                   TEXT,
    emotions_triggered_json       TEXT,
    product_attachability_score   REAL,
    transferability_score         REAL,
    freshness_score               REAL,
    cooked_score                  REAL,
    overall_opportunity_score     REAL,
    hook_mutations_json           TEXT,
    raw_analysis_json             TEXT,
    error                         TEXT
);
CREATE INDEX IF NOT EXISTS idx_reports_created_at ON reports(created_at);
CREATE INDEX IF NOT EXISTS idx_reports_video_id   ON reports(video_id);

-- Phase 6: per-scene-concept user feedback. Powers the taste-feedback
-- loop that conditions Pass 3 in future runs.
CREATE TABLE IF NOT EXISTS concept_feedback (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id            TEXT NOT NULL,
    concept_source_type  TEXT NOT NULL,
    concept_name         TEXT NOT NULL,
    concept_index        INTEGER NOT NULL,
    rating               TEXT NOT NULL,
    note                 TEXT,
    created_at           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_concept_feedback_report_id ON concept_feedback(report_id);
CREATE INDEX IF NOT EXISTS idx_concept_feedback_rating    ON concept_feedback(rating);
CREATE INDEX IF NOT EXISTS idx_concept_feedback_created   ON concept_feedback(created_at);
"""

# Phase 6: allowed ratings for concept_feedback.rating.
ALLOWED_RATINGS: tuple[str, ...] = ("fire", "good", "meh", "cringe", "cooked")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


@contextmanager
def connect(db_path: Path | str) -> Iterator[sqlite3.Connection]:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path | str) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def _dump_json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _load_json(value: Any) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def insert_report(db_path: Path | str, report: dict[str, Any]) -> str:
    """Insert a report row. `report` is a flat dict; this function handles
    JSON encoding of list/dict fields. Returns the generated id."""
    init_db(db_path)
    report_id = report.get("id") or _new_id()
    row = {
        "id": report_id,
        "created_at": report.get("created_at") or _now_iso(),
        "platform": report.get("platform") or "TikTok",
        "source_url": report.get("source_url"),
        "submitted_url": report["submitted_url"],
        "video_id": report.get("video_id"),
        "creator_username": report.get("creator_username"),
        "creator_nickname": report.get("creator_nickname"),
        "caption": report.get("caption"),
        "metrics_json": _dump_json(report.get("metrics")),
        "duration": report.get("duration"),
        "contact_sheet_path": report.get("contact_sheet_path"),
        "cover_url": report.get("cover_url"),
        "video_download_url_saved": 1 if report.get("video_download_url_saved") else 0,
        "apify_run_id": report.get("apify_run_id"),
        "apify_dataset_id": report.get("apify_dataset_id"),
        "apify_usage_usd": report.get("apify_usage_usd"),
        "apify_charged_events_json": _dump_json(report.get("apify_charged_events")),
        "visual_hook_summary": report.get("visual_hook_summary"),
        "onscreen_text": report.get("onscreen_text"),
        "emotional_mechanic": report.get("emotional_mechanic"),
        "viewer_role": report.get("viewer_role"),
        "emotions_triggered_json": _dump_json(report.get("emotions_triggered")),
        "product_attachability_score": report.get("product_attachability_score"),
        "transferability_score": report.get("transferability_score"),
        "freshness_score": report.get("freshness_score"),
        "cooked_score": report.get("cooked_score"),
        "overall_opportunity_score": report.get("overall_opportunity_score"),
        "hook_mutations_json": _dump_json(report.get("hook_mutations")),
        "raw_analysis_json": _dump_json(report.get("raw_analysis")),
        "error": report.get("error"),
    }
    cols = ", ".join(row.keys())
    placeholders = ", ".join(f":{k}" for k in row.keys())
    with connect(db_path) as conn:
        conn.execute(f"INSERT INTO reports ({cols}) VALUES ({placeholders})", row)
    return report_id


def _row_to_report(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    return {
        "id": d["id"],
        "created_at": d["created_at"],
        "platform": d["platform"],
        "source_url": d["source_url"],
        "submitted_url": d["submitted_url"],
        "video_id": d["video_id"],
        "creator_username": d["creator_username"],
        "creator_nickname": d["creator_nickname"],
        "caption": d["caption"],
        "metrics": _load_json(d["metrics_json"]),
        "duration": d["duration"],
        "contact_sheet_path": d["contact_sheet_path"],
        "cover_url": d["cover_url"],
        "video_download_url_saved": bool(d["video_download_url_saved"]),
        "apify_run_id": d["apify_run_id"],
        "apify_dataset_id": d["apify_dataset_id"],
        "apify_usage_usd": d["apify_usage_usd"],
        "apify_charged_events": _load_json(d["apify_charged_events_json"]),
        "visual_hook_summary": d["visual_hook_summary"],
        "onscreen_text": d["onscreen_text"],
        "emotional_mechanic": d["emotional_mechanic"],
        "viewer_role": d["viewer_role"],
        "emotions_triggered": _load_json(d["emotions_triggered_json"]),
        "product_attachability_score": d["product_attachability_score"],
        "transferability_score": d["transferability_score"],
        "freshness_score": d["freshness_score"],
        "cooked_score": d["cooked_score"],
        "overall_opportunity_score": d["overall_opportunity_score"],
        "hook_mutations": _load_json(d["hook_mutations_json"]),
        "raw_analysis": _load_json(d["raw_analysis_json"]),
        "error": d["error"],
    }


def list_reports(db_path: Path | str, limit: int = 50) -> list[dict[str, Any]]:
    init_db(db_path)
    with connect(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM reports ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [_row_to_report(r) for r in cur.fetchall()]


_ANALYSIS_COLUMNS = (
    "visual_hook_summary",
    "onscreen_text",
    "emotional_mechanic",
    "viewer_role",
    "product_attachability_score",
    "transferability_score",
    "freshness_score",
    "cooked_score",
    "overall_opportunity_score",
)

_ANALYSIS_JSON_COLUMNS = {
    "emotions_triggered": "emotions_triggered_json",
    "hook_mutations": "hook_mutations_json",
    "raw_analysis": "raw_analysis_json",
}


def update_report_analysis(
    db_path: Path | str,
    report_id: str,
    analysis_fields: dict[str, Any],
) -> bool:
    """Update the analysis-related columns on a single report row.
    Returns True if the row existed and was updated, False otherwise.

    Only known analysis keys are written; unknown keys are ignored so a
    future schema bump can't accidentally leak free-form fields into the
    DB. `raw_analysis`, `emotions_triggered`, and `hook_mutations` are
    JSON-encoded into their `*_json` columns."""
    init_db(db_path)
    sets: list[str] = []
    params: dict[str, Any] = {"id": report_id}

    for col in _ANALYSIS_COLUMNS:
        if col in analysis_fields:
            sets.append(f"{col} = :{col}")
            params[col] = analysis_fields[col]

    for src_key, db_col in _ANALYSIS_JSON_COLUMNS.items():
        if src_key in analysis_fields:
            sets.append(f"{db_col} = :{db_col}")
            params[db_col] = _dump_json(analysis_fields[src_key])

    if not sets:
        return False  # nothing to do; treat as no-op rather than an error

    sql = f"UPDATE reports SET {', '.join(sets)} WHERE id = :id"
    with connect(db_path) as conn:
        cur = conn.execute(sql, params)
        return cur.rowcount > 0


def get_report(db_path: Path | str, report_id: str) -> dict[str, Any] | None:
    init_db(db_path)
    with connect(db_path) as conn:
        cur = conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,))
        row = cur.fetchone()
        return _row_to_report(row) if row else None


# ============================================================================
# Phase 6: concept_feedback CRUD + taste summary
# ============================================================================


class FeedbackError(ValueError):
    """Raised on invalid feedback input (bad rating, missing fields)."""


def insert_feedback(
    db_path: Path | str,
    report_id: str,
    concept_source_type: str,
    concept_name: str,
    concept_index: int,
    rating: str,
    note: str | None = None,
) -> int:
    """Insert a feedback row. Returns the new id.
    Raises FeedbackError if `rating` is not in ALLOWED_RATINGS."""
    if rating not in ALLOWED_RATINGS:
        raise FeedbackError(
            f"Invalid rating {rating!r}. Allowed: {', '.join(ALLOWED_RATINGS)}"
        )
    if not report_id or not concept_source_type or not concept_name:
        raise FeedbackError("report_id, concept_source_type, and concept_name are required.")
    init_db(db_path)
    row = {
        "report_id": report_id,
        "concept_source_type": concept_source_type,
        "concept_name": concept_name,
        "concept_index": int(concept_index),
        "rating": rating,
        "note": note,
        "created_at": _now_iso(),
    }
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO concept_feedback
                (report_id, concept_source_type, concept_name, concept_index,
                 rating, note, created_at)
            VALUES (:report_id, :concept_source_type, :concept_name, :concept_index,
                    :rating, :note, :created_at)
            """,
            row,
        )
        return int(cur.lastrowid)


def _row_to_feedback(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    return {
        "id": d["id"],
        "report_id": d["report_id"],
        "concept_source_type": d["concept_source_type"],
        "concept_name": d["concept_name"],
        "concept_index": d["concept_index"],
        "rating": d["rating"],
        "note": d["note"],
        "created_at": d["created_at"],
    }


def list_feedback(db_path: Path | str, limit: int = 50) -> list[dict[str, Any]]:
    """Return the most recent feedback rows, newest first."""
    init_db(db_path)
    with connect(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM concept_feedback ORDER BY created_at DESC, id DESC LIMIT ?",
            (int(limit),),
        )
        return [_row_to_feedback(r) for r in cur.fetchall()]


def get_report_feedback(
    db_path: Path | str,
    report_id: str,
) -> list[dict[str, Any]]:
    """All feedback rows for one report, newest first."""
    init_db(db_path)
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            SELECT * FROM concept_feedback
            WHERE report_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (report_id,),
        )
        return [_row_to_feedback(r) for r in cur.fetchall()]


_POSITIVE_RATINGS = ("fire", "good")
_NEGATIVE_RATINGS = ("cringe", "cooked")


def build_taste_summary(
    db_path: Path | str,
    limit: int = 50,
) -> str | None:
    """Build a compact, human-readable taste summary from recent feedback,
    or return None if no feedback exists. The summary is designed to be
    embedded in the Pass 3 system / user prompt as a soft guide."""
    rows = list_feedback(db_path, limit=limit)
    if not rows:
        return None

    liked: list[str] = []
    disliked: list[str] = []
    meh: list[str] = []
    notes: list[str] = []
    for r in rows:
        label = f"{r['concept_name']} ({r['concept_source_type']})"
        if r["rating"] in _POSITIVE_RATINGS:
            liked.append(label)
        elif r["rating"] in _NEGATIVE_RATINGS:
            disliked.append(label)
        else:
            meh.append(label)
        if r.get("note"):
            notes.append(r["note"])

    lines: list[str] = []
    if liked:
        lines.append("User tends to like:")
        for item in liked[:10]:
            lines.append(f"  - {item}")
    if disliked:
        if lines:
            lines.append("")
        lines.append("User dislikes:")
        for item in disliked[:10]:
            lines.append(f"  - {item}")
    if notes:
        if lines:
            lines.append("")
        lines.append("Recent notes:")
        for item in notes[:10]:
            lines.append(f"  - {item}")

    return "\n".join(lines) if lines else None
