"""
Static Client Registry

Manages static client registrations loaded from YAML configuration.
Static clients (like laptop-admin) have permanent keypairs that don't expire.

Configuration format (config/clients.yaml):
```yaml
clients:
  laptop-admin:
    description: "Local laptop scripts with full access"
    public_keys:
      - "base64-encoded-public-key-1"
      - "base64-encoded-public-key-2"  # Can have multiple keys
    scopes: ["*"]
    enabled: true
```
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, FrozenSet
import yaml

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from backend.core.signing.keys import base64_to_public_key
from backend.core.signing.scopes import Scope, parse_scopes
from backend.core.paths import get_config_path

logger = logging.getLogger(__name__)


@dataclass
class StaticClient:
    """
    A statically configured client.
    
    Attributes:
        client_id: Unique identifier (e.g., "laptop-admin")
        description: Human-readable description
        public_keys: List of valid public keys (allows key rotation)
        scopes: Granted permission scopes
        enabled: Whether client is active
    """
    client_id: str
    description: str
    public_keys: List[Ed25519PublicKey]
    scopes: FrozenSet[Scope]
    enabled: bool = True
    
    def has_public_key(self, public_key: Ed25519PublicKey) -> bool:
        """Check if the given public key belongs to this client."""
        # Compare raw bytes since Ed25519PublicKey doesn't have __eq__
        from backend.core.signing.keys import public_key_to_base64
        key_b64 = public_key_to_base64(public_key)
        for pk in self.public_keys:
            if public_key_to_base64(pk) == key_b64:
                return True
        return False


class StaticClientRegistry:
    """
    Registry of statically configured clients.
    
    Loaded from YAML config file at startup.
    Thread-safe for reads (immutable after load).
    """
    
    def __init__(self):
        self._clients: Dict[str, StaticClient] = {}
        self._key_to_client: Dict[str, str] = {}  # base64 key -> client_id
        self._loaded = False
    
    def load_from_yaml(self, config_path: Path) -> None:
        """
        Load client configuration from YAML file.
        
        Args:
            config_path: Path to clients.yaml
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Clients config not found: {config_path}")
            self._loaded = True
            return
        
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        
        clients_config = config.get("clients", {})
        
        for client_id, client_data in clients_config.items():
            try:
                client = self._parse_client(client_id, client_data)
                self._register_client(client)
            except Exception as e:
                logger.error(f"Failed to load client '{client_id}': {e}")
                raise ValueError(f"Invalid client config for '{client_id}': {e}") from e
        
        self._loaded = True
        logger.info(f"Loaded {len(self._clients)} static clients from {config_path}")
    
    def _parse_client(self, client_id: str, data: dict) -> StaticClient:
        """Parse a client configuration entry."""
        description = data.get("description", "")
        enabled = data.get("enabled", True)
        
        # Parse public keys
        key_strings = data.get("public_keys", [])
        public_keys = []
        for key_str in key_strings:
            try:
                pk = base64_to_public_key(key_str)
                public_keys.append(pk)
            except ValueError as e:
                raise ValueError(f"Invalid public key: {e}")
        
        # Parse scopes
        scope_strings = data.get("scopes", [])
        scopes = parse_scopes(scope_strings)
        
        return StaticClient(
            client_id=client_id,
            description=description,
            public_keys=public_keys,
            scopes=scopes,
            enabled=enabled,
        )
    
    def _register_client(self, client: StaticClient) -> None:
        """Add a client to the registry."""
        self._clients[client.client_id] = client
        
        # Index by public key for fast lookup
        from backend.core.signing.keys import public_key_to_base64
        for pk in client.public_keys:
            key_b64 = public_key_to_base64(pk)
            self._key_to_client[key_b64] = client.client_id
    
    def get_client(self, client_id: str) -> Optional[StaticClient]:
        """
        Get a client by ID.
        
        Args:
            client_id: Client identifier
            
        Returns:
            StaticClient if found and enabled, None otherwise
        """
        client = self._clients.get(client_id)
        if client and client.enabled:
            return client
        return None
    
    def get_client_by_key(self, public_key: Ed25519PublicKey) -> Optional[StaticClient]:
        """
        Find client by public key.
        
        Args:
            public_key: Ed25519 public key
            
        Returns:
            StaticClient if found and enabled, None otherwise
        """
        from backend.core.signing.keys import public_key_to_base64
        key_b64 = public_key_to_base64(public_key)
        
        client_id = self._key_to_client.get(key_b64)
        if client_id:
            return self.get_client(client_id)
        return None
    
    def list_clients(self) -> List[str]:
        """Get list of all registered client IDs."""
        return list(self._clients.keys())
    
    @property
    def is_loaded(self) -> bool:
        """Check if registry has been loaded."""
        return self._loaded


# Global registry instance
static_registry = StaticClientRegistry()


def load_static_clients(config_path: Optional[Path] = None) -> StaticClientRegistry:
    """
    Load static clients from configuration.

    Args:
        config_path: Path to clients.yaml. If None, uses get_config_path().

    Returns:
        The loaded registry
    """
    if config_path is None:
        config_path = get_config_path("clients.yaml")
        if config_path is None:
            # No clients.yaml found - signing auth disabled, API key auth only
            logger.info("No clients.yaml found - Ed25519 signing auth disabled")
            return static_registry

    static_registry.load_from_yaml(config_path)
    return static_registry
