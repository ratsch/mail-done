"""
Tests for Document Host Configuration.

Phase 2 tests for:
- HostConfig validation
- FolderScanConfig defaults and overrides
- Configuration loading from YAML
"""

import pytest
import tempfile
from pathlib import Path
import yaml

from backend.core.documents.config import (
    HostConfig,
    FolderScanConfig,
    HostConfigManager,
    get_host_config,
    get_all_host_configs,
    create_scan_config,
)


class TestHostConfig:
    """Tests for HostConfig dataclass."""

    def test_local_host_config(self):
        """Should create valid local host config."""
        config = HostConfig(
            name="localhost",
            type="local",
            mount_point="/",
        )
        assert config.name == "localhost"
        assert config.type == "local"
        assert config.mount_point == "/"
        assert config.read_only is True

    def test_network_mount_config(self):
        """Should create valid network mount config."""
        config = HostConfig(
            name="nas",
            type="network_mount",
            mount_point="/mnt/nas",
        )
        assert config.type == "network_mount"
        assert config.mount_point == "/mnt/nas"

    def test_ssh_host_config(self):
        """Should create valid SSH host config."""
        config = HostConfig(
            name="remote",
            type="ssh",
            ssh_host="server.example.com",
            ssh_user="user",
            ssh_port=22,
        )
        assert config.type == "ssh"
        assert config.ssh_host == "server.example.com"
        assert config.ssh_user == "user"

    def test_invalid_type_raises(self):
        """Should raise error for invalid host type."""
        with pytest.raises(ValueError, match="Invalid host type"):
            HostConfig(
                name="test",
                type="invalid_type",
                mount_point="/",
            )

    def test_ssh_requires_host(self):
        """SSH config should require ssh_host."""
        with pytest.raises(ValueError, match="ssh_host is required"):
            HostConfig(
                name="test",
                type="ssh",
                ssh_user="user",
            )

    def test_ssh_requires_user(self):
        """SSH config should require ssh_user."""
        with pytest.raises(ValueError, match="ssh_user is required"):
            HostConfig(
                name="test",
                type="ssh",
                ssh_host="server.example.com",
            )

    def test_local_requires_mount_point(self):
        """Local config should require mount_point."""
        with pytest.raises(ValueError, match="mount_point is required"):
            HostConfig(
                name="test",
                type="local",
            )

    def test_max_file_size_bytes(self):
        """Should convert MB to bytes correctly."""
        config = HostConfig(
            name="test",
            type="local",
            mount_point="/",
            max_file_size_mb=50,
        )
        assert config.max_file_size_bytes == 50 * 1024 * 1024

    def test_extension_allowed(self):
        """Should check extension correctly."""
        config = HostConfig(
            name="test",
            type="local",
            mount_point="/",
            allowed_extensions=["pdf", "docx", "xlsx"],
        )
        assert config.is_extension_allowed("pdf") is True
        assert config.is_extension_allowed(".pdf") is True
        assert config.is_extension_allowed("PDF") is True
        assert config.is_extension_allowed("exe") is False

    def test_default_extensions(self):
        """Should have sensible default extensions."""
        config = HostConfig(
            name="test",
            type="local",
            mount_point="/",
        )
        assert "pdf" in config.allowed_extensions
        assert "docx" in config.allowed_extensions
        assert "xlsx" in config.allowed_extensions

    def test_default_exclude_patterns(self):
        """Should have sensible default exclusions."""
        config = HostConfig(
            name="test",
            type="local",
            mount_point="/",
        )
        assert "*.tmp" in config.exclude_patterns
        assert ".DS_Store" in config.exclude_patterns


class TestFolderScanConfig:
    """Tests for FolderScanConfig dataclass."""

    @pytest.fixture
    def host(self):
        """Create a test host config."""
        return HostConfig(
            name="test",
            type="local",
            mount_point="/",
            allowed_extensions=["pdf", "docx"],
            exclude_patterns=["*.tmp"],
            max_file_size_mb=100,
        )

    def test_basic_scan_config(self, host):
        """Should create valid scan config."""
        config = FolderScanConfig(
            host=host,
            base_path="/documents",
        )
        assert config.base_path == "/documents"
        assert config.recursive is True
        assert config.skip_hidden is True

    def test_effective_extensions_uses_host_default(self, host):
        """Should use host extensions when not overridden."""
        config = FolderScanConfig(
            host=host,
            base_path="/documents",
        )
        assert config.effective_extensions == ["pdf", "docx"]

    def test_effective_extensions_override(self, host):
        """Should use override extensions when provided."""
        config = FolderScanConfig(
            host=host,
            base_path="/documents",
            extensions=["txt", "md"],
        )
        assert config.effective_extensions == ["txt", "md"]

    def test_effective_exclude_patterns_default(self, host):
        """Should use host patterns when not overridden."""
        config = FolderScanConfig(
            host=host,
            base_path="/documents",
        )
        assert config.effective_exclude_patterns == ["*.tmp"]

    def test_effective_exclude_patterns_override(self, host):
        """Should use override patterns when provided."""
        config = FolderScanConfig(
            host=host,
            base_path="/documents",
            exclude_patterns=["*.bak", "*.old"],
        )
        assert config.effective_exclude_patterns == ["*.bak", "*.old"]

    def test_effective_max_file_size(self, host):
        """Should calculate effective max file size."""
        config = FolderScanConfig(
            host=host,
            base_path="/documents",
        )
        assert config.effective_max_file_size_bytes == 100 * 1024 * 1024

        config_override = FolderScanConfig(
            host=host,
            base_path="/documents",
            max_file_size_mb=50,
        )
        assert config_override.effective_max_file_size_bytes == 50 * 1024 * 1024


class TestHostConfigManager:
    """Tests for HostConfigManager singleton."""

    def test_singleton(self):
        """Should return same instance."""
        manager1 = HostConfigManager()
        manager2 = HostConfigManager()
        assert manager1 is manager2

    def test_default_localhost(self):
        """Should have localhost as default."""
        # Reset singleton for clean test
        HostConfigManager._instance = None
        HostConfigManager._configs = {}

        manager = HostConfigManager()
        localhost = manager.get_host("localhost")

        assert localhost is not None
        assert localhost.type == "local"

    def test_load_from_yaml(self):
        """Should load configs from YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_file = config_dir / "document_hosts.yaml"

            config_data = {
                "hosts": [
                    {
                        "name": "test-host",
                        "type": "local",
                        "mount_point": "/test",
                        "max_file_size_mb": 50,
                    },
                    {
                        "name": "nas",
                        "type": "network_mount",
                        "mount_point": "/mnt/nas",
                    },
                ]
            }

            with open(config_file, 'w') as f:
                yaml.dump(config_data, f)

            # Reset and reload
            HostConfigManager._instance = None
            HostConfigManager._configs = {}

            import backend.core.documents.config as config_module
            original_config_dir = config_module.CONFIG_DIR

            try:
                config_module.CONFIG_DIR = config_dir
                manager = HostConfigManager()

                test_host = manager.get_host("test-host")
                assert test_host is not None
                assert test_host.mount_point == "/test"
                assert test_host.max_file_size_mb == 50

                nas = manager.get_host("nas")
                assert nas is not None
                assert nas.type == "network_mount"

            finally:
                config_module.CONFIG_DIR = original_config_dir
                HostConfigManager._instance = None
                HostConfigManager._configs = {}


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_get_host_config(self):
        """Should return host config by name."""
        # Reset singleton
        HostConfigManager._instance = None
        HostConfigManager._configs = {}

        config = get_host_config("localhost")
        assert config is not None
        assert config.name == "localhost"

    def test_get_host_config_not_found(self):
        """Should return None for unknown host."""
        # Reset singleton
        HostConfigManager._instance = None
        HostConfigManager._configs = {}

        config = get_host_config("nonexistent-host")
        assert config is None

    def test_get_all_host_configs(self):
        """Should return all configs."""
        # Reset singleton
        HostConfigManager._instance = None
        HostConfigManager._configs = {}

        configs = get_all_host_configs()
        assert isinstance(configs, dict)
        assert "localhost" in configs

    def test_create_scan_config(self):
        """Should create scan config from host name."""
        # Reset singleton
        HostConfigManager._instance = None
        HostConfigManager._configs = {}

        config = create_scan_config(
            host_name="localhost",
            base_path="/documents",
            recursive=True,
            extensions=["pdf"],
        )

        assert config.base_path == "/documents"
        assert config.recursive is True
        assert config.extensions == ["pdf"]
        assert config.host.name == "localhost"

    def test_create_scan_config_unknown_host(self):
        """Should raise for unknown host."""
        # Reset singleton
        HostConfigManager._instance = None
        HostConfigManager._configs = {}

        with pytest.raises(ValueError, match="Unknown host"):
            create_scan_config(
                host_name="nonexistent",
                base_path="/documents",
            )
