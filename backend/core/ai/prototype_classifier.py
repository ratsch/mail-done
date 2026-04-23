"""Prototype-based spam classification.

Each spam campaign (BCBS insurance, AAA road kit, …) is represented by
a single centroid vector computed from seed-email embeddings. Emails
whose embedding lies within a configured cosine distance of any centroid
are auto-classified as spam — no LLM call needed.

Pipeline:
  1. ``rebuild_prototypes()`` reads the YAML config, runs the seed regex
     against the DB to pick seed emails, averages their embeddings, and
     UPSERTs the normalized centroid into ``spam_prototype_centroids``.
  2. ``sweep_prototypes()`` runs a single SQL JOIN that matches inbox
     emails against all centroids via pgvector cosine distance (uses the
     existing diskann index), returning rows to act on.

TTL: configurable (default 7 days). A centroid is rebuilt when stale,
when the config hash changes, or on explicit ``--force`` from the
rebuild CLI.

The table stores centroid + metadata; classification itself never
touches Python-side vector math at runtime.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.core.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_TTL_DAYS = 7


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


@dataclass
class PrototypeConfig:
    """One prototype's configuration from spam_prototypes.yaml."""
    name: str
    description: str
    subject_regex: str
    from_regex: Optional[str]
    threshold: float
    min_seed_count: int
    action_folder: str
    action_category: str

    def config_hash(self, embedding_model: str) -> str:
        """Hash of inputs that invalidate the centroid.

        Changing the seed regex, threshold, or embedding model makes
        the stored centroid stale — we force rebuild by detecting hash
        mismatch.
        """
        payload = json.dumps({
            "subject_regex": self.subject_regex,
            "from_regex": self.from_regex or "",
            "threshold": self.threshold,
            "embedding_model": embedding_model,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


def load_prototype_config(config_path: Optional[str] = None) -> Tuple[List[PrototypeConfig], str, int]:
    """Load ``spam_prototypes.yaml`` → list of PrototypeConfig.

    Returns ``(prototypes, embedding_model, ttl_days)``. Resolves the
    path via CONFIG_DIR env / mail-done-config overlay the same way the
    rest of the system does.
    """
    if config_path is None:
        from backend.core.paths import get_config_path  # lazy
        resolved = get_config_path("spam_prototypes.yaml")
        config_path = str(resolved) if resolved else "config/spam_prototypes.yaml"

    path = Path(config_path)
    if not path.exists():
        logger.info(f"No prototype config at {path} — prototype filter inactive")
        return [], DEFAULT_EMBEDDING_MODEL, DEFAULT_TTL_DAYS

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    embedding_model = data.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
    ttl_days = int(data.get("ttl_days", DEFAULT_TTL_DAYS))

    prototypes: List[PrototypeConfig] = []
    for item in data.get("prototypes", []):
        prototypes.append(PrototypeConfig(
            name=item["name"],
            description=item.get("description", ""),
            subject_regex=item["seed_filter"]["subject_regex"],
            from_regex=item["seed_filter"].get("from_regex"),
            threshold=float(item["threshold"]),
            min_seed_count=int(item.get("min_seed_count", 0)),
            action_folder=item["action"]["folder"],
            action_category=item["action"]["category"],
        ))
    logger.info(f"Loaded {len(prototypes)} prototype config(s) from {path}")
    return prototypes, embedding_model, ttl_days


# ---------------------------------------------------------------------------
# Centroid build / rebuild
# ---------------------------------------------------------------------------


def _fetch_seed_embeddings(
    db: Session, cfg: PrototypeConfig, embedding_model: str
) -> Tuple[Optional[np.ndarray], int]:
    """Run the seed-filter regex against the DB, fetch matching embeddings.

    Returns ``(stacked_array[n, D], n)`` where D is the configured embedding
    dimension (EMBEDDING_DIM), or ``(None, 0)`` if no matches.
    """
    # Postgres regex operator is ~*, our patterns are already case-insensitive.
    # We OR the subject and from matches (from is optional).
    params: Dict[str, Any] = {
        "subj_re": cfg.subject_regex,
        "embedding_model": embedding_model,
    }
    where_from = ""
    if cfg.from_regex:
        where_from = " OR e.from_address ~ :from_re"
        params["from_re"] = cfg.from_regex

    rows = db.execute(text(f"""
        SELECT ee.embedding::text
        FROM emails e
        JOIN email_embeddings ee ON ee.email_id = e.id
        WHERE ee.embedding_model = :embedding_model
          AND (COALESCE(e.subject, '') ~ :subj_re {where_from})
    """), params).fetchall()

    if not rows:
        return None, 0

    expected_dim = get_settings().embedding_dim
    vecs = []
    for (emb_str,) in rows:
        if not emb_str:
            continue
        v = np.fromstring(emb_str.strip("[]"), sep=",", dtype=np.float32)
        if v.shape == (expected_dim,):
            vecs.append(v)
    if not vecs:
        return None, 0

    arr = np.vstack(vecs)
    # L2-normalize each vector so the mean is a valid anchor for cosine search
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / (norms + 1e-9)
    return arr, len(vecs)


def _compute_centroid(arr: np.ndarray) -> np.ndarray:
    """Normalized mean of a (n, d) array of unit vectors."""
    m = arr.mean(axis=0)
    m = m / (np.linalg.norm(m) + 1e-9)
    return m.astype(np.float32)


def _centroid_to_pgvector(v: np.ndarray) -> str:
    """Serialize a numpy vector as pgvector literal: '[0.123,0.456,...]'."""
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def needs_rebuild(
    existing: Optional[Dict[str, Any]], cfg: PrototypeConfig,
    embedding_model: str, ttl_days: int, now: Optional[datetime] = None,
) -> Tuple[bool, str]:
    """Should this prototype be rebuilt? Returns (yes/no, reason)."""
    if existing is None:
        return True, "not yet built"
    now = now or datetime.utcnow()
    built_at = existing["built_at"]
    if isinstance(built_at, str):
        built_at = datetime.fromisoformat(built_at)
    age = now - built_at
    if age > timedelta(days=ttl_days):
        return True, f"stale ({age.days}d > {ttl_days}d TTL)"
    if existing.get("config_hash") != cfg.config_hash(embedding_model):
        return True, "config hash changed"
    if existing.get("embedding_model") != embedding_model:
        return True, "embedding model changed"
    return False, f"fresh ({age.days}d old)"


def rebuild_one(
    db: Session, cfg: PrototypeConfig, embedding_model: str, force: bool = False,
    ttl_days: int = DEFAULT_TTL_DAYS,
) -> Dict[str, Any]:
    """Rebuild or refresh one prototype's centroid. Commits the session."""
    existing = db.execute(text("""
        SELECT name, built_at, config_hash, embedding_model, seed_count
        FROM spam_prototype_centroids WHERE name = :name
    """), {"name": cfg.name}).mappings().first()

    if not force:
        rebuild, reason = needs_rebuild(existing, cfg, embedding_model, ttl_days)
        if not rebuild:
            logger.info(f"[{cfg.name}] skipping rebuild: {reason}")
            return {"name": cfg.name, "rebuilt": False, "reason": reason,
                    "seed_count": existing["seed_count"] if existing else 0}
    else:
        reason = "forced"

    arr, n = _fetch_seed_embeddings(db, cfg, embedding_model)
    if arr is None or n < cfg.min_seed_count:
        logger.warning(
            f"[{cfg.name}] skipping rebuild: only {n} seeds found "
            f"(min required: {cfg.min_seed_count})"
        )
        return {"name": cfg.name, "rebuilt": False,
                "reason": f"only {n} seeds < min {cfg.min_seed_count}", "seed_count": n}

    centroid = _compute_centroid(arr)
    centroid_str = _centroid_to_pgvector(centroid)

    db.execute(text("""
        INSERT INTO spam_prototype_centroids
            (name, description, centroid, threshold, action_folder, action_category,
             seed_count, built_at, config_hash, embedding_model)
        VALUES (:name, :description, CAST(:centroid AS vector), :threshold,
                :action_folder, :action_category, :seed_count, now(),
                :config_hash, :embedding_model)
        ON CONFLICT (name) DO UPDATE SET
            description = EXCLUDED.description,
            centroid = EXCLUDED.centroid,
            threshold = EXCLUDED.threshold,
            action_folder = EXCLUDED.action_folder,
            action_category = EXCLUDED.action_category,
            seed_count = EXCLUDED.seed_count,
            built_at = EXCLUDED.built_at,
            config_hash = EXCLUDED.config_hash,
            embedding_model = EXCLUDED.embedding_model
    """), {
        "name": cfg.name,
        "description": cfg.description,
        "centroid": centroid_str,
        "threshold": cfg.threshold,
        "action_folder": cfg.action_folder,
        "action_category": cfg.action_category,
        "seed_count": n,
        "config_hash": cfg.config_hash(embedding_model),
        "embedding_model": embedding_model,
    })
    db.commit()
    logger.info(f"[{cfg.name}] rebuilt: {n} seeds, threshold {cfg.threshold}, reason {reason!r}")
    return {"name": cfg.name, "rebuilt": True, "reason": reason, "seed_count": n}


def rebuild_all(
    db: Session, config_path: Optional[str] = None, force: bool = False,
) -> List[Dict[str, Any]]:
    """Rebuild all prototypes from config. Returns per-prototype status."""
    prototypes, embedding_model, ttl_days = load_prototype_config(config_path)
    return [rebuild_one(db, cfg, embedding_model, force=force, ttl_days=ttl_days)
            for cfg in prototypes]


# ---------------------------------------------------------------------------
# Sweep (classification)
# ---------------------------------------------------------------------------


@dataclass
class PrototypeMatch:
    """One email matched against a prototype."""
    email_id: str
    uid: str
    account_id: str
    folder: str
    message_id: str
    subject: str
    from_address: str
    prototype_name: str
    distance: float
    action_folder: str
    action_category: str


def find_matches(
    db: Session, account_id: Optional[str] = None, folder: str = "INBOX",
    since_days: int = 30, limit: int = 500,
) -> List[PrototypeMatch]:
    """Find INBOX emails whose embedding is within a prototype's threshold.

    One SQL JOIN against ``spam_prototype_centroids`` × ``email_embeddings``,
    using pgvector's diskann index. Excludes emails already categorized as
    spam (so we don't move them twice). Rows returned are ready to act on.
    """
    # DISTINCT ON picks the closest prototype per email when more than one
    # matches — we only want to move the email once.
    params = {
        "folder": folder,
        "since_days": since_days,
        "limit": limit,
    }
    account_filter = ""
    if account_id:
        account_filter = "AND e.account_id = :account_id"
        params["account_id"] = account_id

    rows = db.execute(text(f"""
        SELECT DISTINCT ON (e.id)
            e.id::text AS email_id,
            e.uid,
            e.account_id,
            e.folder,
            e.message_id,
            e.subject,
            e.from_address,
            p.name AS prototype_name,
            (ee.embedding <=> p.centroid) AS distance,
            p.action_folder,
            p.action_category
        FROM emails e
        JOIN email_embeddings ee ON ee.email_id = e.id
        CROSS JOIN spam_prototype_centroids p
        LEFT JOIN email_metadata em ON em.email_id = e.id
        WHERE e.folder = :folder
          {account_filter}
          AND e.date >= NOW() - make_interval(days => :since_days)
          AND ee.embedding_model = p.embedding_model
          AND (ee.embedding <=> p.centroid) < p.threshold
          AND (em.ai_category IS NULL OR em.ai_category != p.action_category)
        ORDER BY e.id, (ee.embedding <=> p.centroid)
        LIMIT :limit
    """), params).mappings().all()

    return [PrototypeMatch(**dict(r)) for r in rows]
