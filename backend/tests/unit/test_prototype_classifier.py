"""Unit tests for prototype-based spam classification.

Covers:
- Config loading (YAML → PrototypeConfig)
- config_hash stability / sensitivity to inputs
- needs_rebuild() TTL + hash + not-yet-built logic
- Centroid computation (L2-norm + mean + re-norm)
- pgvector serialization
- SQL query semantics: build correct WHERE clauses
  (real pgvector operations are tested live via scripts/rebuild_spam_prototypes.py;
   here we mock the DB layer to test our Python logic)
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.core.ai.prototype_classifier import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TTL_DAYS,
    PrototypeConfig,
    PrototypeMatch,
    _centroid_to_pgvector,
    _compute_centroid,
    _fetch_seed_embeddings,
    find_matches,
    load_prototype_config,
    needs_rebuild,
    rebuild_one,
)


# ---------------------------------------------------------------------------
# PrototypeConfig + config_hash
# ---------------------------------------------------------------------------


class TestConfigHash:
    """The config hash drives rebuild-on-change: identical inputs must
    produce identical hashes, and every tracked input must change the hash."""

    def _make(self, **overrides) -> PrototypeConfig:
        base = dict(
            name="test_proto",
            description="...",
            subject_regex="(?i)bcbs",
            from_regex="(?i)blue",
            threshold=0.20,
            min_seed_count=10,
            action_folder="MD/Spam",
            action_category="spam",
        )
        base.update(overrides)
        return PrototypeConfig(**base)

    def test_hash_is_deterministic(self):
        cfg = self._make()
        assert cfg.config_hash("text-embedding-3-large") == cfg.config_hash("text-embedding-3-large")

    def test_hash_changes_on_subject_regex(self):
        a = self._make().config_hash("m")
        b = self._make(subject_regex="(?i)other").config_hash("m")
        assert a != b

    def test_hash_changes_on_from_regex(self):
        a = self._make().config_hash("m")
        b = self._make(from_regex="(?i)other").config_hash("m")
        assert a != b

    def test_hash_changes_on_threshold(self):
        a = self._make().config_hash("m")
        b = self._make(threshold=0.30).config_hash("m")
        assert a != b

    def test_hash_changes_on_embedding_model(self):
        cfg = self._make()
        assert cfg.config_hash("model-a") != cfg.config_hash("model-b")

    def test_hash_invariant_to_description(self):
        """description is just metadata — should not force a rebuild."""
        a = self._make(description="old").config_hash("m")
        b = self._make(description="new").config_hash("m")
        assert a == b

    def test_hash_invariant_to_min_seed_count(self):
        """min_seed_count is a gating threshold, not centroid-shaping."""
        a = self._make(min_seed_count=10).config_hash("m")
        b = self._make(min_seed_count=100).config_hash("m")
        assert a == b

    def test_hash_handles_empty_from_regex(self):
        cfg = self._make(from_regex=None)
        h = cfg.config_hash("m")
        assert len(h) == 64  # sha256 hex


# ---------------------------------------------------------------------------
# load_prototype_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_missing_file_returns_empty(self, tmp_path):
        """No config → empty prototype list, sensible defaults."""
        missing = tmp_path / "does-not-exist.yaml"
        prototypes, model, ttl = load_prototype_config(str(missing))
        assert prototypes == []
        assert model == DEFAULT_EMBEDDING_MODEL
        assert ttl == DEFAULT_TTL_DAYS

    def test_loads_valid_yaml(self, tmp_path):
        yaml_file = tmp_path / "spam_prototypes.yaml"
        yaml_file.write_text("""
ttl_days: 14
embedding_model: text-embedding-3-large
prototypes:
  - name: test_spam
    description: "Test prototype"
    seed_filter:
      subject_regex: "(?i)spammy"
      from_regex: "(?i)spammer"
    threshold: 0.15
    min_seed_count: 50
    action:
      folder: "MD/Spam"
      category: spam
""")
        prototypes, model, ttl = load_prototype_config(str(yaml_file))
        assert len(prototypes) == 1
        p = prototypes[0]
        assert p.name == "test_spam"
        assert p.threshold == 0.15
        assert p.min_seed_count == 50
        assert p.action_folder == "MD/Spam"
        assert p.action_category == "spam"
        assert model == "text-embedding-3-large"
        assert ttl == 14

    def test_from_regex_optional(self, tmp_path):
        """from_regex is allowed to be absent — only subject_regex is required."""
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text("""
prototypes:
  - name: subj_only
    seed_filter:
      subject_regex: "(?i)foo"
    threshold: 0.2
    action:
      folder: "MD/Spam"
      category: spam
""")
        prototypes, _, _ = load_prototype_config(str(yaml_file))
        assert prototypes[0].from_regex is None

    def test_empty_prototypes_list(self, tmp_path):
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("prototypes: []\n")
        prototypes, _, _ = load_prototype_config(str(yaml_file))
        assert prototypes == []

    def test_multiple_prototypes_loaded_in_order(self, tmp_path):
        yaml_file = tmp_path / "two.yaml"
        yaml_file.write_text("""
prototypes:
  - name: first
    seed_filter:
      subject_regex: "1"
    threshold: 0.1
    action:
      folder: "F1"
      category: spam
  - name: second
    seed_filter:
      subject_regex: "2"
    threshold: 0.2
    action:
      folder: "F2"
      category: marketing
""")
        prototypes, _, _ = load_prototype_config(str(yaml_file))
        assert [p.name for p in prototypes] == ["first", "second"]


# ---------------------------------------------------------------------------
# needs_rebuild
# ---------------------------------------------------------------------------


class TestNeedsRebuild:
    def _cfg(self, **kw) -> PrototypeConfig:
        base = dict(
            name="p", description="", subject_regex="x", from_regex=None,
            threshold=0.2, min_seed_count=0, action_folder="MD/Spam",
            action_category="spam",
        )
        base.update(kw)
        return PrototypeConfig(**base)

    def test_nonexistent_triggers_rebuild(self):
        cfg = self._cfg()
        yes, reason = needs_rebuild(None, cfg, "m", 7)
        assert yes is True
        assert "not yet built" in reason

    def test_fresh_within_ttl_no_rebuild(self):
        cfg = self._cfg()
        existing = {
            "built_at": datetime.utcnow() - timedelta(days=3),
            "config_hash": cfg.config_hash("m"),
            "embedding_model": "m",
        }
        yes, reason = needs_rebuild(existing, cfg, "m", 7)
        assert yes is False
        assert "fresh" in reason

    def test_stale_past_ttl_rebuilds(self):
        cfg = self._cfg()
        existing = {
            "built_at": datetime.utcnow() - timedelta(days=10),
            "config_hash": cfg.config_hash("m"),
            "embedding_model": "m",
        }
        yes, reason = needs_rebuild(existing, cfg, "m", 7)
        assert yes is True
        assert "stale" in reason

    def test_config_hash_mismatch_rebuilds(self):
        cfg = self._cfg()
        existing = {
            "built_at": datetime.utcnow(),
            "config_hash": "different_hash",
            "embedding_model": "m",
        }
        yes, reason = needs_rebuild(existing, cfg, "m", 7)
        assert yes is True
        assert "config hash" in reason

    def test_embedding_model_change_rebuilds(self):
        """Centroids are model-specific — model change invalidates them."""
        cfg = self._cfg()
        existing = {
            "built_at": datetime.utcnow(),
            "config_hash": cfg.config_hash("old-model"),
            "embedding_model": "old-model",
        }
        yes, reason = needs_rebuild(existing, cfg, "new-model", 7)
        assert yes is True
        # Either 'config hash' (because the hash includes model) or 'embedding model'
        # will match — both are valid reasons.
        assert "model" in reason or "hash" in reason

    def test_isoformat_built_at_string_parsed(self):
        """PG sometimes hands us datetimes as strings depending on driver."""
        cfg = self._cfg()
        existing = {
            "built_at": (datetime.utcnow() - timedelta(days=2)).isoformat(),
            "config_hash": cfg.config_hash("m"),
            "embedding_model": "m",
        }
        yes, _ = needs_rebuild(existing, cfg, "m", 7)
        assert yes is False


# ---------------------------------------------------------------------------
# Centroid computation
# ---------------------------------------------------------------------------


class TestCentroid:
    def test_compute_centroid_returns_unit_vector(self):
        rng = np.random.default_rng(42)
        arr = rng.standard_normal((10, 3072)).astype(np.float32)
        # Input should already be normalized — do it explicitly
        arr /= np.linalg.norm(arr, axis=1, keepdims=True)
        m = _compute_centroid(arr)
        assert m.shape == (3072,)
        assert m.dtype == np.float32
        # Unit-length (within float tolerance)
        assert abs(np.linalg.norm(m) - 1.0) < 1e-5

    def test_centroid_of_identical_vectors_is_that_vector(self):
        """Edge case: mean of n copies of v / |v| = v / |v|."""
        v = np.ones(3072, dtype=np.float32) / np.sqrt(3072)
        arr = np.stack([v] * 5)
        m = _compute_centroid(arr)
        np.testing.assert_allclose(m, v, atol=1e-5)

    def test_centroid_of_opposite_vectors_normalizes_safely(self):
        """Degenerate case: v + (-v) = 0; the normalization uses +1e-9 epsilon
        so we don't divide by zero. The resulting vector can be arbitrary
        direction but must not be NaN / inf."""
        v = np.ones(3072, dtype=np.float32) / np.sqrt(3072)
        arr = np.stack([v, -v])
        m = _compute_centroid(arr)
        assert not np.isnan(m).any()
        assert not np.isinf(m).any()

    def test_pgvector_serialization_format(self):
        v = np.array([1.0, 0.5, -0.25], dtype=np.float32)
        s = _centroid_to_pgvector(v)
        # pgvector expects '[a,b,c]' with no spaces
        assert s.startswith("[")
        assert s.endswith("]")
        parts = s[1:-1].split(",")
        assert parts == ["1.000000", "0.500000", "-0.250000"]

    def test_pgvector_serialization_roundtrip(self):
        """Serialized string + numpy.fromstring gets us back the original
        (within 6-decimal precision)."""
        v = np.array([0.123456, -0.789012, 1e-6], dtype=np.float32)
        s = _centroid_to_pgvector(v)
        back = np.fromstring(s.strip("[]"), sep=",", dtype=np.float32)
        np.testing.assert_allclose(back, v, atol=1e-5)


# ---------------------------------------------------------------------------
# _fetch_seed_embeddings (mocked DB)
# ---------------------------------------------------------------------------


class TestFetchSeedEmbeddings:
    def test_no_matches_returns_none(self):
        db = MagicMock()
        db.execute.return_value.fetchall.return_value = []
        cfg = PrototypeConfig(
            name="p", description="", subject_regex="x", from_regex=None,
            threshold=0.2, min_seed_count=0, action_folder="MD/Spam",
            action_category="spam",
        )
        arr, n = _fetch_seed_embeddings(db, cfg, "m")
        assert arr is None
        assert n == 0

    def test_fetches_and_normalizes_embeddings(self):
        # Two 3072-dim vectors, pgvector text format
        v1 = np.ones(3072, dtype=np.float32) * 2.0  # |v1| = 2 * sqrt(3072)
        v2 = np.ones(3072, dtype=np.float32) * 0.5  # |v2| = 0.5 * sqrt(3072)
        s1 = "[" + ",".join(f"{x:.6f}" for x in v1) + "]"
        s2 = "[" + ",".join(f"{x:.6f}" for x in v2) + "]"

        db = MagicMock()
        db.execute.return_value.fetchall.return_value = [(s1,), (s2,)]

        cfg = PrototypeConfig(
            name="p", description="", subject_regex="x", from_regex=None,
            threshold=0.2, min_seed_count=0, action_folder="MD/Spam",
            action_category="spam",
        )
        arr, n = _fetch_seed_embeddings(db, cfg, "m")
        assert n == 2
        assert arr.shape == (2, 3072)
        # Rows must be L2-normalized (unit vectors)
        norms = np.linalg.norm(arr, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-5)

    def test_skips_malformed_embeddings(self):
        """Any vector whose serialized form doesn't parse to 3072 dims is dropped."""
        bad = "[1.0, 2.0, 3.0]"  # only 3 dims
        good = np.ones(3072, dtype=np.float32)
        good_str = "[" + ",".join(f"{x:.6f}" for x in good) + "]"

        db = MagicMock()
        db.execute.return_value.fetchall.return_value = [(bad,), (good_str,)]
        cfg = PrototypeConfig(
            name="p", description="", subject_regex="x", from_regex=None,
            threshold=0.2, min_seed_count=0, action_folder="MD/Spam",
            action_category="spam",
        )
        arr, n = _fetch_seed_embeddings(db, cfg, "m")
        assert n == 1
        assert arr.shape == (1, 3072)

    def test_from_regex_included_in_where_when_set(self):
        """When from_regex is set, the SQL should match either subject OR from."""
        db = MagicMock()
        db.execute.return_value.fetchall.return_value = []
        cfg = PrototypeConfig(
            name="p", description="", subject_regex="sub_re",
            from_regex="from_re",
            threshold=0.2, min_seed_count=0, action_folder="MD/Spam",
            action_category="spam",
        )
        _fetch_seed_embeddings(db, cfg, "m")
        args, kwargs = db.execute.call_args
        sql = str(args[0])
        params = args[1] if len(args) > 1 else kwargs.get('params', {})
        assert "e.from_address ~" in sql  # OR branch included
        assert params.get("subj_re") == "sub_re"
        assert params.get("from_re") == "from_re"

    def test_from_regex_absent_skips_from_branch(self):
        db = MagicMock()
        db.execute.return_value.fetchall.return_value = []
        cfg = PrototypeConfig(
            name="p", description="", subject_regex="sub_re",
            from_regex=None,
            threshold=0.2, min_seed_count=0, action_folder="MD/Spam",
            action_category="spam",
        )
        _fetch_seed_embeddings(db, cfg, "m")
        args, _ = db.execute.call_args
        sql = str(args[0])
        assert "e.from_address ~" not in sql


# ---------------------------------------------------------------------------
# rebuild_one
# ---------------------------------------------------------------------------


class TestRebuildOne:
    def _cfg(self, min_seed=0) -> PrototypeConfig:
        return PrototypeConfig(
            name="p", description="", subject_regex="x", from_regex=None,
            threshold=0.2, min_seed_count=min_seed, action_folder="MD/Spam",
            action_category="spam",
        )

    def test_skips_rebuild_when_fresh(self):
        cfg = self._cfg()
        db = MagicMock()
        # Mock existing fresh row
        mock_row = {
            "name": "p",
            "built_at": datetime.utcnow() - timedelta(days=1),
            "config_hash": cfg.config_hash("m"),
            "embedding_model": "m",
            "seed_count": 50,
        }
        db.execute.return_value.mappings.return_value.first.return_value = mock_row

        result = rebuild_one(db, cfg, "m", force=False, ttl_days=7)
        assert result["rebuilt"] is False
        assert result["seed_count"] == 50
        assert "fresh" in result["reason"]
        # No INSERT/UPSERT should have been issued — only the existence SELECT
        # Actual call count: 1 (the SELECT for existing row)
        assert db.execute.call_count == 1

    def test_force_bypasses_ttl(self):
        """With force=True, the freshness check is skipped."""
        cfg = self._cfg()

        db = MagicMock()
        # A "fresh" existing row that would normally skip
        mock_exists = {
            "name": "p",
            "built_at": datetime.utcnow() - timedelta(hours=1),
            "config_hash": cfg.config_hash("m"),
            "embedding_model": "m",
            "seed_count": 5,
        }
        # Make embeddings fetch return one vector
        v = np.ones(3072, dtype=np.float32) / np.sqrt(3072)
        vec_str = "[" + ",".join(f"{x:.6f}" for x in v) + "]"

        # Three sequential .execute() calls:
        # 1. SELECT existing row
        # 2. SELECT embeddings
        # 3. INSERT ... ON CONFLICT UPDATE
        first_call = MagicMock()
        first_call.mappings.return_value.first.return_value = mock_exists
        second_call = MagicMock()
        second_call.fetchall.return_value = [(vec_str,)]
        third_call = MagicMock()
        db.execute.side_effect = [first_call, second_call, third_call]

        result = rebuild_one(db, cfg, "m", force=True, ttl_days=7)
        assert result["rebuilt"] is True
        assert result["reason"] == "forced"
        assert result["seed_count"] == 1
        # Commit must be called for the upsert
        assert db.commit.called

    def test_below_min_seed_count_skips(self):
        """If fewer seeds than min_seed_count, we skip — avoids noisy centroids."""
        cfg = self._cfg(min_seed=10)
        db = MagicMock()

        first_call = MagicMock()
        first_call.mappings.return_value.first.return_value = None  # no existing
        v = np.ones(3072, dtype=np.float32) / np.sqrt(3072)
        vec_str = "[" + ",".join(f"{x:.6f}" for x in v) + "]"
        second_call = MagicMock()
        second_call.fetchall.return_value = [(vec_str,), (vec_str,)]  # only 2 seeds
        db.execute.side_effect = [first_call, second_call]

        result = rebuild_one(db, cfg, "m", force=False, ttl_days=7)
        assert result["rebuilt"] is False
        assert result["seed_count"] == 2
        assert "< min" in result["reason"]
        # No upsert should have happened
        assert not db.commit.called


# ---------------------------------------------------------------------------
# find_matches (query shape)
# ---------------------------------------------------------------------------


class TestFindMatches:
    def test_no_matches_returns_empty(self):
        db = MagicMock()
        db.execute.return_value.mappings.return_value.all.return_value = []
        result = find_matches(db, account_id="work", folder="INBOX",
                              since_days=7, limit=100)
        assert result == []

    def test_account_filter_added_when_set(self):
        db = MagicMock()
        db.execute.return_value.mappings.return_value.all.return_value = []
        find_matches(db, account_id="work", folder="INBOX", since_days=7, limit=100)
        args, _ = db.execute.call_args
        sql = str(args[0])
        params = args[1]
        assert "e.account_id = :account_id" in sql
        assert params["account_id"] == "work"

    def test_account_filter_omitted_when_none(self):
        db = MagicMock()
        db.execute.return_value.mappings.return_value.all.return_value = []
        find_matches(db, account_id=None, folder="INBOX", since_days=7, limit=100)
        args, _ = db.execute.call_args
        sql = str(args[0])
        params = args[1]
        assert "e.account_id = :account_id" not in sql
        assert "account_id" not in params

    def test_parameters_passed_correctly(self):
        db = MagicMock()
        db.execute.return_value.mappings.return_value.all.return_value = []
        find_matches(db, account_id="work", folder="MD/Inbox",
                     since_days=14, limit=50)
        _, _ = db.execute.call_args
        params = db.execute.call_args[0][1]
        assert params["folder"] == "MD/Inbox"
        assert params["since_days"] == 14
        assert params["limit"] == 50

    def test_returns_prototype_match_objects(self):
        db = MagicMock()
        fake_row = {
            "email_id": "e-123",
            "uid": "42",
            "account_id": "work",
            "folder": "INBOX",
            "message_id": "<abc@example.com>",
            "subject": "Test",
            "from_address": "spam@throwaway.com",
            "prototype_name": "bcbs_insurance_spam",
            "distance": 0.123,
            "action_folder": "MD/Spam",
            "action_category": "spam",
        }
        db.execute.return_value.mappings.return_value.all.return_value = [fake_row]
        result = find_matches(db, account_id="work", folder="INBOX",
                              since_days=7, limit=100)
        assert len(result) == 1
        m = result[0]
        assert isinstance(m, PrototypeMatch)
        assert m.prototype_name == "bcbs_insurance_spam"
        assert m.distance == 0.123
        assert m.action_folder == "MD/Spam"

    def test_sql_excludes_already_spam(self):
        """The query must NOT re-flag emails already categorized as spam
        (prevents redundant moves)."""
        db = MagicMock()
        db.execute.return_value.mappings.return_value.all.return_value = []
        find_matches(db, account_id="work", folder="INBOX", since_days=7, limit=100)
        sql = str(db.execute.call_args[0][0])
        assert "em.ai_category IS NULL OR em.ai_category != p.action_category" in sql

    def test_sql_uses_diskann_operator(self):
        """The SQL must use pgvector's cosine-distance operator '<=>' so the
        diskann index is exercised. A mistake here (e.g., L2 '<->') would
        silently fall back to a sequential scan and be orders of magnitude
        slower on production volume."""
        db = MagicMock()
        db.execute.return_value.mappings.return_value.all.return_value = []
        find_matches(db, account_id="work", folder="INBOX", since_days=7, limit=100)
        sql = str(db.execute.call_args[0][0])
        assert "<=>" in sql
