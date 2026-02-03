"""
Account Manager - Multi-Account Configuration Management

Manages multiple email account configurations with secure credential handling.
Loads account metadata from accounts.yaml and credentials from environment variables.
"""
from typing import Dict, Optional, List
import yaml
import os
from pathlib import Path
from pydantic import BaseModel, Field
import logging

from backend.core.paths import get_config_path

logger = logging.getLogger(__name__)


class AccountConfig(BaseModel):
    """Single email account configuration"""
    nickname: str
    display_name: str
    imap_host: str
    imap_port: int = 993
    imap_use_ssl: bool = True
    imap_username: Optional[str] = None  # Loaded from env
    imap_password: Optional[str] = None  # Loaded from env
    
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_use_tls: bool = True
    smtp_username: Optional[str] = None  # Loaded from env
    smtp_password: Optional[str] = None  # Loaded from env
    
    # Authentication type: "password" or "oauth2"
    auth_type: str = "password"
    
    # OAuth2 settings (for Microsoft 365 / Outlook)
    oauth2_tenant_id: Optional[str] = None
    oauth2_client_id: Optional[str] = None
    oauth2_client_secret: Optional[str] = None  # Loaded from env (for app-only auth)
    oauth2_refresh_token: Optional[str] = None  # Loaded from env (for delegated auth)
    
    allow_moves_to: List[str] = Field(default_factory=list)
    folders: Dict[str, str] = Field(default_factory=dict)
    route_inbox_to: Optional[str] = None  # Route INBOX emails to this account after processing
    default_move_target: Optional[str] = None  # Default target account for moves without explicit target_account

    def can_move_to(self, target: str) -> bool:
        """Check if moves to target account are allowed"""
        return target in self.allow_moves_to


class AccountManager:
    """
    Manages multiple email account configurations.
    
    Loads account metadata from config/accounts.yaml and credentials from environment variables.
    Falls back to legacy single-account mode if accounts.yaml doesn't exist.
    
    Usage:
        manager = AccountManager()
        account = manager.get_account('work')
        imap_config = manager.get_imap_config('work')
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize account manager.

        Args:
            config_path: Path to accounts.yaml configuration file.
                        If None, uses get_config_path() to resolve.
        """
        if config_path is None:
            resolved = get_config_path("accounts.yaml")
            self.config_path = resolved if resolved else Path("config/accounts.yaml")
        else:
            self.config_path = Path(config_path)
        self.accounts: Dict[str, AccountConfig] = {}
        self.default_account: Optional[str] = None
        self.settings: Dict = {}

        # Load configuration
        self._load_config()
        self._load_credentials()
        self._validate_accounts()
    
    def _load_config(self):
        """Load account configurations from YAML file"""
        if not self.config_path.exists():
            logger.warning(f"Config file {self.config_path} not found, using legacy single-account mode")
            self._create_legacy_account()
            return
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if not config:
                logger.warning("Empty config file, using legacy mode")
                self._create_legacy_account()
                return
            
            # Load each account
            for nickname, acc_config in config.get('accounts', {}).items():
                try:
                    # Get OAuth2 settings if present
                    oauth2_config = acc_config.get('oauth2', {})
                    auth_type = acc_config.get('auth_type', 'password')
                    
                    account = AccountConfig(
                        nickname=nickname,
                        display_name=acc_config.get('display_name', nickname),
                        imap_host=acc_config['imap']['host'],
                        imap_port=acc_config['imap'].get('port', 993),
                        imap_use_ssl=acc_config['imap'].get('use_ssl', True),
                        smtp_host=acc_config.get('smtp', {}).get('host'),
                        smtp_port=acc_config.get('smtp', {}).get('port', 587),
                        smtp_use_tls=acc_config.get('smtp', {}).get('use_tls', True),
                        auth_type=auth_type,
                        oauth2_tenant_id=oauth2_config.get('tenant_id'),
                        oauth2_client_id=oauth2_config.get('client_id'),
                        allow_moves_to=acc_config.get('allow_moves_to', []),
                        folders=acc_config.get('folders', {}),
                        route_inbox_to=acc_config.get('route_inbox_to'),
                        default_move_target=acc_config.get('default_move_target')
                    )
                    self.accounts[nickname] = account
                    auth_info = f"[{auth_type}]" if auth_type == "oauth2" else ""
                    logger.info(f"Loaded account configuration: {nickname} ({account.display_name}) {auth_info}")
                except Exception as e:
                    logger.error(f"Failed to load account '{nickname}': {e}")
                    continue
            
            # Load global settings
            self.settings = config.get('settings', {})
            self.default_account = self.settings.get('default_account', 'work')
            
            logger.info(f"Loaded {len(self.accounts)} account(s), default: {self.default_account}")
            
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            self._create_legacy_account()
    
    def _load_credentials(self):
        """Load credentials from environment variables"""
        for nickname, account in self.accounts.items():
            # Sanitize nickname: uppercase, replace dashes/underscores with underscores
            sanitized_nickname = nickname.upper().replace('-', '_').replace(' ', '_')
            
            # Always load username (needed for both auth types)
            if not account.imap_username:
                imap_user_env = f'IMAP_USERNAME_{sanitized_nickname}'
                account.imap_username = os.getenv(imap_user_env)
            
            if account.auth_type == "oauth2":
                # OAuth2 authentication - load OAuth2 credentials
                oauth2_secret_env = f'OAUTH2_CLIENT_SECRET_{sanitized_nickname}'
                oauth2_refresh_env = f'OAUTH2_REFRESH_TOKEN_{sanitized_nickname}'
                
                account.oauth2_client_secret = os.getenv(oauth2_secret_env)
                account.oauth2_refresh_token = os.getenv(oauth2_refresh_env)
                
                # Validate OAuth2 credentials
                if not account.imap_username:
                    logger.error(f"Missing username for OAuth2 account '{nickname}'")
                    logger.error(f"  Expected: IMAP_USERNAME_{sanitized_nickname}")
                
                if not account.oauth2_refresh_token and not account.oauth2_client_secret:
                    logger.error(f"Missing OAuth2 credentials for account '{nickname}'")
                    logger.error(f"  Expected: OAUTH2_REFRESH_TOKEN_{sanitized_nickname} (delegated auth)")
                    logger.error(f"       or: OAUTH2_CLIENT_SECRET_{sanitized_nickname} (app-only auth)")
                
                if not account.oauth2_tenant_id or not account.oauth2_client_id:
                    logger.error(f"Missing OAuth2 config for account '{nickname}'")
                    logger.error(f"  Required in accounts.yaml: oauth2.tenant_id, oauth2.client_id")
            else:
                # Password authentication - load password
                if not account.imap_password:
                    imap_pass_env = f'IMAP_PASSWORD_{sanitized_nickname}'
                    account.imap_password = os.getenv(imap_pass_env)
                
                # SMTP credentials (often same as IMAP)
                smtp_user_env = f'SMTP_USERNAME_{sanitized_nickname}'
                smtp_pass_env = f'SMTP_PASSWORD_{sanitized_nickname}'
                
                account.smtp_username = os.getenv(smtp_user_env) or account.imap_username
                account.smtp_password = os.getenv(smtp_pass_env) or account.imap_password
                
                # Validate password credentials exist
                if not account.imap_username or not account.imap_password:
                    logger.error(f"Missing IMAP credentials for account '{nickname}'")
                    logger.error(f"  Expected: IMAP_USERNAME_{sanitized_nickname}, IMAP_PASSWORD_{sanitized_nickname}")
    
    def _validate_accounts(self):
        """Validate account configurations on startup"""
        if not self.accounts:
            raise ValueError("No accounts configured! Check config/accounts.yaml and environment variables.")
        
        # Check for missing credentials
        invalid_accounts = []
        for nickname, account in self.accounts.items():
            # Skip validation for legacy work account (may not have credentials yet)
            if nickname == 'work' and not account.imap_username:
                logger.debug(f"Skipping credential validation for legacy 'work' account")
                continue
            
            # Validate based on auth type
            if account.auth_type == "oauth2":
                # OAuth2 requires: username, tenant_id, client_id, and either refresh_token or client_secret
                has_oauth2_creds = (
                    account.imap_username and
                    account.oauth2_tenant_id and
                    account.oauth2_client_id and
                    (account.oauth2_refresh_token or account.oauth2_client_secret)
                )
                if not has_oauth2_creds:
                    invalid_accounts.append(f"{nickname} (oauth2)")
            else:
                # Password auth requires username and password
                if not account.imap_username or not account.imap_password:
                    invalid_accounts.append(f"{nickname} (password)")
        
        if invalid_accounts:
            raise ValueError(
                f"Missing credentials for accounts: {', '.join(invalid_accounts)}\n"
                f"For password auth: IMAP_USERNAME_<ACCOUNT>, IMAP_PASSWORD_<ACCOUNT>\n"
                f"For OAuth2: IMAP_USERNAME_<ACCOUNT>, OAUTH2_REFRESH_TOKEN_<ACCOUNT> (or OAUTH2_CLIENT_SECRET_<ACCOUNT>)"
            )
        
        # Check default account exists
        if self.default_account and self.default_account not in self.accounts:
            logger.warning(f"Default account '{self.default_account}' not found, using first available")
            self.default_account = list(self.accounts.keys())[0]
        
        logger.info(f"✅ Account validation passed: {len(self.accounts)} account(s) ready")
    
    def _create_legacy_account(self):
        """Create 'work' account from legacy environment variables"""
        logger.info("Creating legacy 'work' account from IMAP_* environment variables")
        
        account = AccountConfig(
            nickname='work',
            display_name='Work Account',
            imap_host=os.getenv('IMAP_HOST', 'imap.gmail.com'),
            imap_port=int(os.getenv('IMAP_PORT', 993)),
            imap_username=os.getenv('IMAP_USERNAME'),
            imap_password=os.getenv('IMAP_PASSWORD'),
            smtp_host=os.getenv('SMTP_HOST', 'smtp.gmail.com'),
            smtp_port=int(os.getenv('SMTP_PORT', 587)),
            smtp_username=os.getenv('SMTP_USERNAME') or os.getenv('IMAP_USERNAME'),
            smtp_password=os.getenv('SMTP_PASSWORD') or os.getenv('IMAP_PASSWORD')
        )
        
        self.accounts['work'] = account
        self.default_account = 'work'
        
        # Skip credential validation for legacy mode (credentials may not be set yet)
        # They'll be validated when actually used
    
    def get_account(self, nickname: Optional[str] = None) -> AccountConfig:
        """
        Get account configuration by nickname.
        
        Args:
            nickname: Account nickname (e.g., 'work', 'personal').
                     If None, returns default account.
        
        Returns:
            AccountConfig for the specified account
            
        Raises:
            ValueError: If account not found
        """
        if nickname is None:
            nickname = self.default_account
        
        if nickname not in self.accounts:
            available = ', '.join(self.accounts.keys())
            raise ValueError(
                f"Unknown account: '{nickname}'\n"
                f"Available accounts: {available}"
            )
        
        return self.accounts[nickname]
    
    def get_imap_config(self, nickname: Optional[str] = None):
        """
        Get IMAPConfig for an account.
        
        Args:
            nickname: Account nickname. If None, uses default.
            
        Returns:
            IMAPConfig instance ready for IMAPMonitor
        """
        from backend.core.email.imap_monitor import IMAPConfig
        
        account = self.get_account(nickname)
        return IMAPConfig(
            host=account.imap_host,
            username=account.imap_username,
            port=account.imap_port,
            use_ssl=account.imap_use_ssl,
            auth_type=account.auth_type,
            password=account.imap_password,
            oauth2_tenant_id=account.oauth2_tenant_id,
            oauth2_client_id=account.oauth2_client_id,
            oauth2_client_secret=account.oauth2_client_secret,
            oauth2_refresh_token=account.oauth2_refresh_token
        )
    
    def can_move_between(self, from_account: str, to_account: str) -> bool:
        """
        Check if cross-account move is allowed.
        
        Args:
            from_account: Source account nickname
            to_account: Target account nickname
            
        Returns:
            True if move is allowed by whitelist
        """
        try:
            from_acc = self.get_account(from_account)
            return from_acc.can_move_to(to_account)
        except ValueError:
            return False
    
    def list_accounts(self) -> List[str]:
        """Get list of all configured account nicknames"""
        return list(self.accounts.keys())
    
    def get_account_display_info(self, nickname: str) -> Dict[str, str]:
        """
        Get account display information (safe to show in UI/logs).
        
        Returns dict with: nickname, display_name, email, host
        """
        account = self.get_account(nickname)
        return {
            'nickname': nickname,
            'display_name': account.display_name,
            'email': account.imap_username,
            'host': account.imap_host,
            'allow_moves_to': account.allow_moves_to
        }
    
    def get_setting(self, key: str, default=None):
        """Get a global setting value"""
        return self.settings.get(key, default)


