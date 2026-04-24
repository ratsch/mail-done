"""Unit tests for AccountManager.resolve_folder_role().

Covers the folder-role-token resolution added in cf1dc81
(`{junk}` / `{trash}` / `{archive}` / `{inbox}` / `{sent}`):
- token → per-account override
- token → global Settings fallback (junk/trash/archive)
- token → built-in IMAP defaults (inbox/sent)
- literal path passthrough
- unresolvable token raises
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.core.accounts.manager import AccountConfig, AccountManager


def _mgr(accounts: dict[str, dict]) -> AccountManager:
    """Build an AccountManager without touching YAML / env.

    Bypasses AccountManager.__init__ so we don't need a real accounts.yaml
    or live credentials. Tests only exercise ``resolve_folder_role``, which
    depends solely on ``self.accounts``.

    ``AccountConfig`` requires ``nickname``, ``display_name``, and
    ``imap_host``; the rest have defaults. We fill placeholders so tests
    stay focused on folder-role behavior.
    """
    mgr = AccountManager.__new__(AccountManager)
    mgr.accounts = {
        nick: AccountConfig(
            nickname=nick,
            display_name=nick,
            imap_host="imap.test.example",
            **conf,
        )
        for nick, conf in accounts.items()
    }
    mgr.settings = {}
    mgr.default_account = next(iter(mgr.accounts), None)
    return mgr


class TestResolveFolderRolePerAccount:
    def test_per_account_override_wins(self):
        mgr = _mgr({
            "personal": {"folders": {"junk": "Junk"}},
            "gmail":    {"folders": {"junk": "[Gmail]/Spam"}},
        })
        assert mgr.resolve_folder_role("personal", "{junk}") == "Junk"
        assert mgr.resolve_folder_role("gmail", "{junk}") == "[Gmail]/Spam"

    def test_gmail_style_sent_folder(self):
        mgr = _mgr({"gmail": {"folders": {"sent": "[Gmail]/Sent Mail"}}})
        assert mgr.resolve_folder_role("gmail", "{sent}") == "[Gmail]/Sent Mail"


class TestResolveFolderRoleFallbacks:
    def test_inbox_default_for_account_without_folders(self):
        """inbox has a universal IMAP default, so missing folders.inbox
        falls back to "INBOX" — no Settings needed."""
        mgr = _mgr({"acct": {"folders": {}}})
        assert mgr.resolve_folder_role("acct", "{inbox}") == "INBOX"

    def test_sent_default_for_account_without_folders(self):
        mgr = _mgr({"acct": {"folders": {}}})
        assert mgr.resolve_folder_role("acct", "{sent}") == "Sent"

    def test_junk_falls_back_to_settings(self):
        mgr = _mgr({"acct": {"folders": {}}})
        fake_settings = MagicMock(folder_junk="MD/Spam",
                                  folder_trash="Trash",
                                  folder_archive="Archive")
        with patch("backend.core.config.get_settings", return_value=fake_settings):
            assert mgr.resolve_folder_role("acct", "{junk}") == "MD/Spam"
            assert mgr.resolve_folder_role("acct", "{trash}") == "Trash"
            assert mgr.resolve_folder_role("acct", "{archive}") == "Archive"


class TestResolveFolderRolePassthrough:
    def test_literal_path_passes_through_unchanged(self):
        mgr = _mgr({"acct": {"folders": {"junk": "Junk"}}})
        assert mgr.resolve_folder_role("acct", "MD/Spam") == "MD/Spam"
        assert mgr.resolve_folder_role("acct", "Archive/2024") == "Archive/2024"

    def test_empty_string_passes_through(self):
        mgr = _mgr({"acct": {"folders": {}}})
        assert mgr.resolve_folder_role("acct", "") == ""

    def test_partial_token_is_not_resolved(self):
        """Only ``{role}`` exactly — ``{role}/sub`` is a literal."""
        mgr = _mgr({"acct": {"folders": {"junk": "Junk"}}})
        assert mgr.resolve_folder_role("acct", "{junk}/sub") == "{junk}/sub"


class TestResolveFolderRoleUnresolvable:
    def test_unknown_role_raises(self):
        mgr = _mgr({"acct": {"folders": {}}})
        fake_settings = MagicMock(folder_junk="MD/Spam",
                                  folder_trash="Trash",
                                  folder_archive="Archive")
        with patch("backend.core.config.get_settings", return_value=fake_settings):
            with pytest.raises(ValueError, match="Cannot resolve folder role"):
                mgr.resolve_folder_role("acct", "{nonsense}")

    def test_unknown_account_falls_back_to_global(self):
        """A missing account_id is OK as long as the role has a global
        fallback — resolve should still succeed."""
        mgr = _mgr({})
        fake_settings = MagicMock(folder_junk="MD/Spam",
                                  folder_trash="Trash",
                                  folder_archive="Archive")
        with patch("backend.core.config.get_settings", return_value=fake_settings):
            assert mgr.resolve_folder_role("ghost", "{junk}") == "MD/Spam"
            assert mgr.resolve_folder_role("ghost", "{inbox}") == "INBOX"

    def test_settings_unavailable_still_resolves_inbox(self):
        """If Settings fails to load, inbox/sent defaults still work."""
        mgr = _mgr({"acct": {"folders": {}}})
        with patch("backend.core.config.get_settings",
                   side_effect=RuntimeError("settings blew up")):
            assert mgr.resolve_folder_role("acct", "{inbox}") == "INBOX"
            assert mgr.resolve_folder_role("acct", "{sent}") == "Sent"

    def test_settings_unavailable_and_role_without_default_raises(self):
        """Junk has no built-in default — Settings failure makes it
        unresolvable and must raise."""
        mgr = _mgr({"acct": {"folders": {}}})
        with patch("backend.core.config.get_settings",
                   side_effect=RuntimeError("settings blew up")):
            with pytest.raises(ValueError, match="Cannot resolve folder role"):
                mgr.resolve_folder_role("acct", "{junk}")


class TestPipelineResolveFolder:
    """_resolve_folder helper on EmailProcessingPipeline — the chokepoint
    that guarantees no raw ``{role}`` token ever reaches IMAP."""

    def _pipeline_with(self, account_manager):
        """Build a bare EmailProcessingPipeline instance without running
        __init__ (which would require IMAP/DB credentials)."""
        from process_inbox import EmailProcessingPipeline
        p = EmailProcessingPipeline.__new__(EmailProcessingPipeline)
        p.account_manager = account_manager
        p.account_id = "acct"
        return p

    def test_literal_passes_through_without_manager(self):
        p = self._pipeline_with(account_manager=None)
        assert p._resolve_folder("acct", "MD/Spam") == "MD/Spam"
        assert p._resolve_folder("acct", None) is None
        assert p._resolve_folder("acct", "") == ""

    def test_token_without_manager_raises(self):
        """Without AccountManager, a token MUST raise — silently passing
        '{junk}' to IMAP would create a literal '{junk}' folder."""
        p = self._pipeline_with(account_manager=None)
        with pytest.raises(RuntimeError, match="accounts.yaml is not loaded"):
            p._resolve_folder("acct", "{junk}")

    def test_token_with_manager_delegates(self):
        mgr = _mgr({"acct": {"folders": {"junk": "Junk"}}})
        p = self._pipeline_with(account_manager=mgr)
        assert p._resolve_folder("acct", "{junk}") == "Junk"
