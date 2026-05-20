"""Direct Apify REST client for clockworks/tiktok-video-scraper.

Why direct REST instead of the apify-client SDK: keeps deps tiny, makes
behavior trivially mockable in tests, and the call surface we need is
small (start actor → wait → fetch dataset items).
"""

from __future__ import annotations

import time
from typing import Any, Iterable
from urllib.parse import quote

import requests

from .config import APIFY_ACTOR_ID, APIFY_RUN_INPUT_DEFAULTS
from .models import ApifyRunInfo, NormalizedItem

APIFY_BASE = "https://api.apify.com/v2"
DEFAULT_TIMEOUT_SEC = 60
DEFAULT_POLL_INTERVAL_SEC = 4
DEFAULT_RUN_BUDGET_SEC = 600


class ApifyError(RuntimeError):
    pass


class ApifyClient:
    def __init__(
        self,
        token: str,
        session: requests.Session | None = None,
        actor_id: str = APIFY_ACTOR_ID,
        timeout: int = DEFAULT_TIMEOUT_SEC,
    ):
        if not token:
            raise ApifyError("Apify token is empty.")
        self.token = token
        self.actor_id = actor_id
        self.timeout = timeout
        self.session = session or requests.Session()

    def _params(self, **extra: Any) -> dict[str, Any]:
        p = {"token": self.token}
        p.update(extra)
        return p

    def run_actor(
        self,
        post_urls: list[str],
        poll_interval: int = DEFAULT_POLL_INTERVAL_SEC,
        budget_sec: int = DEFAULT_RUN_BUDGET_SEC,
    ) -> tuple[list[dict[str, Any]], ApifyRunInfo]:
        """Start the actor, wait for completion, return (dataset_items, run_info).
        Raises ApifyError on failure."""
        payload = dict(APIFY_RUN_INPUT_DEFAULTS)
        payload["postURLs"] = list(post_urls)

        actor_path = quote(self.actor_id, safe="")
        start_url = f"{APIFY_BASE}/acts/{actor_path}/runs"
        r = self.session.post(
            start_url,
            params=self._params(),
            json=payload,
            timeout=self.timeout,
        )
        if not r.ok:
            raise ApifyError(f"Failed to start actor: {r.status_code} {r.text[:300]}")
        run = r.json().get("data", {})
        run_id = run.get("id")
        if not run_id:
            raise ApifyError(f"Actor start returned no run id: {run}")

        run = self._wait_for_run(run_id, poll_interval, budget_sec)
        status = run.get("status")
        if status != "SUCCEEDED":
            raise ApifyError(
                f"Apify run {run_id} ended with status={status}. "
                f"stats={run.get('stats')}"
            )

        dataset_id = run.get("defaultDatasetId")
        items = self._fetch_dataset_items(dataset_id) if dataset_id else []

        usage_total_usd = None
        try:
            usage_total_usd = float(run.get("usageTotalUsd")) if run.get("usageTotalUsd") is not None else None
        except (TypeError, ValueError):
            usage_total_usd = None

        charged_events = run.get("chargedEventCounts") or run.get("pricingInfo") or None

        info = ApifyRunInfo(
            run_id=run_id,
            dataset_id=dataset_id,
            usage_total_usd=usage_total_usd,
            charged_events=charged_events,
        )
        return items, info

    def _wait_for_run(self, run_id: str, poll_interval: int, budget_sec: int) -> dict[str, Any]:
        url = f"{APIFY_BASE}/actor-runs/{run_id}"
        deadline = time.monotonic() + budget_sec
        while True:
            r = self.session.get(url, params=self._params(), timeout=self.timeout)
            if not r.ok:
                raise ApifyError(f"Failed to poll run {run_id}: {r.status_code} {r.text[:300]}")
            run = r.json().get("data", {})
            status = run.get("status")
            if status in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"):
                return run
            if time.monotonic() > deadline:
                raise ApifyError(f"Apify run {run_id} exceeded budget ({budget_sec}s). Last status={status}.")
            time.sleep(poll_interval)

    def _fetch_dataset_items(self, dataset_id: str) -> list[dict[str, Any]]:
        url = f"{APIFY_BASE}/datasets/{dataset_id}/items"
        r = self.session.get(
            url,
            params=self._params(clean="true", format="json"),
            timeout=self.timeout,
        )
        if not r.ok:
            raise ApifyError(f"Failed to fetch dataset {dataset_id}: {r.status_code} {r.text[:300]}")
        data = r.json()
        return data if isinstance(data, list) else []


# ---- Normalization ---------------------------------------------------------

def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_item(item: dict[str, Any], submitted_url: str) -> NormalizedItem:
    """Map a raw Apify item to our internal NormalizedItem shape.

    If no playable video_download_url can be found, the returned item has
    .error set; the caller should persist a report row with that error and
    move on (no exceptions for one bad item among many)."""
    media_urls = item.get("mediaUrls") or []
    video_meta = item.get("videoMeta") or {}
    author_meta = item.get("authorMeta") or {}

    video_download_url: str | None = None
    if isinstance(media_urls, list) and media_urls:
        first = media_urls[0]
        if isinstance(first, str) and first:
            video_download_url = first
    if not video_download_url:
        candidate = video_meta.get("downloadAddr")
        if isinstance(candidate, str) and candidate:
            video_download_url = candidate

    metrics = {
        "views": _coerce_int(item.get("playCount")),
        "likes": _coerce_int(item.get("diggCount")),
        "comments": _coerce_int(item.get("commentCount")),
        "shares": _coerce_int(item.get("shareCount")),
        "saves": _coerce_int(item.get("collectCount")),
    }

    norm = NormalizedItem(
        platform="TikTok",
        source_url=item.get("webVideoUrl"),
        submitted_url=submitted_url,
        video_download_url=video_download_url,
        cover_url=video_meta.get("coverUrl") or video_meta.get("cover"),
        creator_username=author_meta.get("name"),
        creator_nickname=author_meta.get("nickName") or author_meta.get("nickname"),
        caption=item.get("text"),
        video_id=str(item["id"]) if item.get("id") is not None else None,
        duration=_coerce_float(video_meta.get("duration")),
        metrics=metrics,
        raw=item,
    )
    if not video_download_url:
        norm.error = "No downloadable video URL in Apify item (mediaUrls and videoMeta.downloadAddr empty)."
    return norm


def normalize_items(
    items: Iterable[dict[str, Any]],
    submitted_urls: list[str],
) -> list[NormalizedItem]:
    """Pair items with submitted URLs. Match by webVideoUrl when possible,
    otherwise pull from unmatched items in submission order. Any URL with
    no item gets an error-only NormalizedItem for traceability."""
    items_list = list(items)
    used: set[int] = set()
    by_url: dict[str, dict[str, Any]] = {}
    for it in items_list:
        wvu = it.get("webVideoUrl")
        if isinstance(wvu, str):
            by_url.setdefault(wvu, it)

    out: list[NormalizedItem] = []
    for url in submitted_urls:
        match = by_url.get(url)
        if match is None or id(match) in used:
            match = next(
                (it for it in items_list if id(it) not in used),
                None,
            )
        if match is None:
            out.append(NormalizedItem(
                platform="TikTok",
                source_url=None,
                submitted_url=url,
                video_download_url=None,
                cover_url=None,
                creator_username=None,
                creator_nickname=None,
                caption=None,
                video_id=None,
                duration=None,
                metrics={"views": None, "likes": None, "comments": None, "shares": None, "saves": None},
                raw={},
                error="No Apify item returned for this URL.",
            ))
            continue
        used.add(id(match))
        out.append(normalize_item(match, submitted_url=url))
    return out
