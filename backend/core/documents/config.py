"""
Document Host Configuration

Defines configuration for accessing documents from different hosts:
- Local filesystem
- Network mounts (NFS, SMB)
- SSH/SCP remote access

Configuration is loaded from config/document_hosts.yaml.
"""

import os
import logging
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

logger = logging.getLogger(__name__)


def _get_stable_machine_id() -> str:
    """
    Get a stable, SSH-accessible machine identifier.

    Uses LOCAL_MACHINE_NAME environment variable if set, otherwise falls back
    to socket.gethostname(). Set LOCAL_MACHINE_NAME to your preferred
    SSH-accessible hostname (e.g., Tailscale name).

    Returns a hostname suitable for SSH access and document retrieval.
    """
    # Check for user-configured machine name
    env_name = os.getenv("LOCAL_MACHINE_NAME")
    if env_name:
        return env_name

    # Fallback to hostname
    hostname = socket.gethostname()
    # Clean up hostname (remove .local suffix on macOS)
    if hostname.endswith(".local"):
        hostname = hostname[:-6]
    return hostname

# Config directory (same as other config files)
CONFIG_DIR = Path(os.getenv("CONFIG_DIR", "config"))


@dataclass
class HostConfig:
    """Configuration for a document host."""

    name: str
    type: str  # 'local', 'network_mount', 'ssh'

    # For local/network_mount
    mount_point: Optional[str] = None

    # For SSH access
    ssh_host: Optional[str] = None
    ssh_user: Optional[str] = None
    ssh_port: int = 22
    ssh_key_path: Optional[str] = None

    # Access settings
    read_only: bool = True
    max_file_size_mb: int = 100
    allowed_extensions: list[str] = field(default_factory=lambda: [
        "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx",
        "txt", "rtf", "odt", "ods", "odp", "csv", "md",
        "jpg", "jpeg", "png", "gif", "tiff", "bmp",
        "eml", "msg"
    ])

    # Exclusion patterns (glob patterns)
    exclude_patterns: list[str] = field(default_factory=lambda: [
        "*.tmp", "*.temp", "*.bak", "*.swp", "*~",
        ".DS_Store", "Thumbs.db", ".git/*", ".svn/*",
        "__pycache__/*", "*.pyc", "node_modules/*",
        ".Trash/*", "$RECYCLE.BIN/*"
    ])

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_types = ('local', 'network_mount', 'ssh')
        if self.type not in valid_types:
            raise ValueError(f"Invalid host type '{self.type}'. Must be one of: {valid_types}")

        if self.type == 'ssh':
            if not self.ssh_host:
                raise ValueError("ssh_host is required for SSH host type")
            if not self.ssh_user:
                raise ValueError("ssh_user is required for SSH host type")

        if self.type in ('local', 'network_mount') and not self.mount_point:
            raise ValueError(f"mount_point is required for {self.type} host type")

    @property
    def max_file_size_bytes(self) -> int:
        """Return max file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024

    def is_extension_allowed(self, extension: str) -> bool:
        """Check if file extension is allowed."""
        ext = extension.lower().lstrip('.')
        return ext in [e.lower().lstrip('.') for e in self.allowed_extensions]


@dataclass
class FolderScanConfig:
    """Configuration for folder scanning operations."""

    host: HostConfig
    base_path: str
    recursive: bool = True
    skip_hidden: bool = True
    follow_symlinks: bool = False

    # Overrides for this specific scan
    extensions: Optional[list[str]] = None
    exclude_patterns: Optional[list[str]] = None
    max_file_size_mb: Optional[int] = None

    @property
    def effective_extensions(self) -> list[str]:
        """Get effective extensions list (override or host default)."""
        return self.extensions if self.extensions else self.host.allowed_extensions

    @property
    def effective_exclude_patterns(self) -> list[str]:
        """Get effective exclude patterns (override or host default)."""
        return self.exclude_patterns if self.exclude_patterns else self.host.exclude_patterns

    @property
    def effective_max_file_size_bytes(self) -> int:
        """Get effective max file size in bytes."""
        mb = self.max_file_size_mb if self.max_file_size_mb else self.host.max_file_size_mb
        return mb * 1024 * 1024


class HostConfigManager:
    """Manager for loading and caching host configurations."""

    _instance: Optional["HostConfigManager"] = None
    _configs: Dict[str, HostConfig] = {}

    def __new__(cls):
        """Singleton pattern for config manager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._configs = {}
            cls._instance._load_configs()
        return cls._instance

    def _load_configs(self):
        """Load host configurations from YAML file."""
        config_path = CONFIG_DIR / "document_hosts.yaml"

        if not config_path.exists():
            hostname = _get_stable_machine_id()
            logger.debug(f"No host config at {config_path}, using hostname '{hostname}'")
            # Use actual hostname for local origins, with "localhost" as alias
            self._configs[hostname] = HostConfig(
                name=hostname,
                type="local",
                mount_point="/"
            )
            # Also register under "localhost" alias for backward compatibility
            self._configs["localhost"] = self._configs[hostname]
            return

        try:
            with open(config_path) as f:
                data = yaml.safe_load(f)

            if not data or "hosts" not in data:
                hostname = _get_stable_machine_id()
                logger.warning(f"Invalid host config file (missing 'hosts' key), using hostname '{hostname}'")
                self._configs[hostname] = HostConfig(
                    name=hostname,
                    type="local",
                    mount_point="/"
                )
                self._configs["localhost"] = self._configs[hostname]
                return

            for host_data in data.get("hosts", []):
                try:
                    config = HostConfig(**host_data)
                    self._configs[config.name] = config
                    logger.info(f"Loaded host config: {config.name} ({config.type})")
                except (TypeError, ValueError) as e:
                    logger.error(f"Invalid host config: {host_data.get('name', 'unknown')}: {e}")

        except yaml.YAMLError as e:
            logger.error(f"Failed to parse host config: {e}")
            hostname = _get_stable_machine_id()
            self._configs[hostname] = HostConfig(
                name=hostname,
                type="local",
                mount_point="/"
            )
            self._configs["localhost"] = self._configs[hostname]

    def get_host(self, name: str) -> Optional[HostConfig]:
        """Get host configuration by name."""
        return self._configs.get(name)

    def get_all_hosts(self) -> Dict[str, HostConfig]:
        """Get all host configurations."""
        return self._configs.copy()

    def reload(self):
        """Reload configurations from disk."""
        self._configs.clear()
        self._load_configs()


def get_host_config(host_name: str) -> Optional[HostConfig]:
    """
    Get host configuration by name.

    This is the main entry point for getting host configurations.

    Args:
        host_name: Name of the host to get config for

    Returns:
        HostConfig if found, None otherwise
    """
    manager = HostConfigManager()
    return manager.get_host(host_name)


def get_all_host_configs() -> Dict[str, HostConfig]:
    """
    Get all configured hosts.

    Returns:
        Dict mapping host names to HostConfig objects
    """
    manager = HostConfigManager()
    return manager.get_all_hosts()


def create_scan_config(
    host_name: str,
    base_path: str,
    recursive: bool = True,
    extensions: Optional[list[str]] = None,
    exclude_patterns: Optional[list[str]] = None,
    max_file_size_mb: Optional[int] = None,
) -> FolderScanConfig:
    """
    Create a folder scan configuration.

    Args:
        host_name: Name of the host to scan
        base_path: Path to scan on the host
        recursive: Whether to scan recursively
        extensions: Override for allowed extensions
        exclude_patterns: Override for exclude patterns
        max_file_size_mb: Override for max file size

    Returns:
        FolderScanConfig ready for use

    Raises:
        ValueError: If host not found
    """
    host = get_host_config(host_name)
    if not host:
        raise ValueError(f"Unknown host: {host_name}")

    return FolderScanConfig(
        host=host,
        base_path=base_path,
        recursive=recursive,
        extensions=extensions,
        exclude_patterns=exclude_patterns,
        max_file_size_mb=max_file_size_mb,
    )
