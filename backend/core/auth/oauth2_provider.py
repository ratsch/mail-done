"""
OAuth2 Token Provider for Microsoft 365 / Outlook

Handles OAuth2 authentication for IMAP/SMTP access to Microsoft 365 accounts.
Supports both:
- Delegated auth (user login with refresh token)
- App-only auth (service account with client credentials)

Usage:
    config = OAuth2Config(
        tenant_id="your-tenant-or-common",
        client_id="your-app-client-id",
        client_secret="your-secret",  # For app-only
        # OR
        refresh_token="user-refresh-token"  # For delegated
    )
    provider = OAuth2Provider(config)
    access_token = provider.get_access_token(username="user@example.com")
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class OAuth2Config:
    """OAuth2 configuration for Microsoft 365"""
    tenant_id: str  # Azure AD tenant ID or "common" / "organizations"
    client_id: str  # Azure AD application (client) ID
    
    # For app-only authentication (service principal)
    client_secret: Optional[str] = None
    
    # For delegated authentication (user login)
    refresh_token: Optional[str] = None
    
    # Token cache file (optional, for persistent caching)
    token_cache_file: Optional[str] = None
    
    # Microsoft Graph/Office365 IMAP scope
    # For delegated: https://outlook.office365.com/IMAP.AccessAsUser.All
    # For app-only: https://outlook.office365.com/.default
    # Note: offline_access is implicit for public clients
    scopes: list = field(default_factory=lambda: [
        "https://outlook.office365.com/IMAP.AccessAsUser.All",
        "https://outlook.office365.com/SMTP.Send",
    ])


class OAuth2Provider:
    """
    OAuth2 token provider for Microsoft 365.
    
    Handles token acquisition, caching, and refresh.
    """
    
    def __init__(self, config: OAuth2Config):
        """
        Initialize OAuth2 provider.
        
        Args:
            config: OAuth2 configuration
        """
        self.config = config
        self._token_cache: Dict[str, dict] = {}  # username -> {access_token, expires_at}
        self._msal_app = None
        
        # Load cached tokens if available
        if config.token_cache_file:
            self._load_token_cache()
    
    def _get_msal_app(self):
        """Get or create MSAL application instance"""
        if self._msal_app is None:
            try:
                import msal
            except ImportError:
                raise ImportError(
                    "MSAL library required for OAuth2 authentication. "
                    "Install with: uv add msal"
                )
            
            authority = f"https://login.microsoftonline.com/{self.config.tenant_id}"
            
            if self.config.client_secret:
                # Confidential client (with client secret)
                self._msal_app = msal.ConfidentialClientApplication(
                    client_id=self.config.client_id,
                    client_credential=self.config.client_secret,
                    authority=authority
                )
            else:
                # Public client (for refresh token flow without secret)
                self._msal_app = msal.PublicClientApplication(
                    client_id=self.config.client_id,
                    authority=authority
                )
        
        return self._msal_app
    
    def get_access_token(self, username: str) -> str:
        """
        Get a valid access token for the given user.
        
        Checks cache first, refreshes if expired.
        
        Args:
            username: Email address (e.g., user@university.edu)
            
        Returns:
            Valid access token string
            
        Raises:
            RuntimeError: If token acquisition fails
        """
        # Check cache
        cached = self._token_cache.get(username)
        if cached and cached.get('expires_at', 0) > time.time() + 60:  # 60s buffer
            logger.debug(f"Using cached access token for {username}")
            return cached['access_token']
        
        # Acquire new token
        logger.info(f"Acquiring new OAuth2 access token for {username}")
        
        app = self._get_msal_app()
        result = None
        
        if self.config.refresh_token:
            # Delegated auth: use refresh token
            result = self._acquire_token_by_refresh_token(app, username)
        elif self.config.client_secret:
            # App-only auth: use client credentials
            result = self._acquire_token_by_client_credentials(app)
        else:
            raise RuntimeError(
                "OAuth2 requires either refresh_token (delegated) or "
                "client_secret (app-only) authentication"
            )
        
        if 'error' in result:
            error_desc = result.get('error_description', result.get('error', 'Unknown error'))
            logger.error(f"OAuth2 token acquisition failed: {error_desc}")
            raise RuntimeError(f"OAuth2 authentication failed: {error_desc}")
        
        access_token = result['access_token']
        expires_in = result.get('expires_in', 3600)
        
        # Cache the token
        self._token_cache[username] = {
            'access_token': access_token,
            'expires_at': time.time() + expires_in
        }
        
        # Update refresh token if a new one was provided
        if 'refresh_token' in result:
            self.config.refresh_token = result['refresh_token']
            logger.info(f"Received new refresh token for {username}")
        
        # Save cache
        if self.config.token_cache_file:
            self._save_token_cache()
        
        logger.info(f"Successfully acquired OAuth2 token for {username} (expires in {expires_in}s)")
        return access_token
    
    def _acquire_token_by_refresh_token(self, app, username: str) -> dict:
        """Acquire token using refresh token (delegated auth)"""
        # MSAL's acquire_token_by_refresh_token is for confidential clients
        # For public clients, we need to use a different approach
        
        try:
            # Try confidential client method first
            if hasattr(app, 'acquire_token_by_refresh_token'):
                result = app.acquire_token_by_refresh_token(
                    refresh_token=self.config.refresh_token,
                    scopes=self.config.scopes
                )
            else:
                # Fallback for public client - use silent acquisition with account
                accounts = app.get_accounts(username=username)
                if accounts:
                    result = app.acquire_token_silent(
                        scopes=self.config.scopes,
                        account=accounts[0]
                    )
                else:
                    # No cached account, need to use refresh token directly
                    # This requires an HTTP call since MSAL public client doesn't expose refresh token method
                    result = self._refresh_token_direct(username)
            
            return result or {'error': 'No result from token acquisition'}
            
        except Exception as e:
            logger.error(f"Error acquiring token by refresh token: {e}")
            return {'error': str(e)}
    
    def _refresh_token_direct(self, username: str) -> dict:
        """Direct refresh token call using HTTP (for public client)"""
        import httpx
        
        token_url = f"https://login.microsoftonline.com/{self.config.tenant_id}/oauth2/v2.0/token"
        
        data = {
            'client_id': self.config.client_id,
            'grant_type': 'refresh_token',
            'refresh_token': self.config.refresh_token,
            'scope': ' '.join(self.config.scopes)
        }
        
        # Add client secret if available
        if self.config.client_secret:
            data['client_secret'] = self.config.client_secret
        
        try:
            with httpx.Client() as client:
                response = client.post(token_url, data=data)
                result = response.json()
                
                if response.status_code != 200:
                    return {
                        'error': result.get('error', 'token_error'),
                        'error_description': result.get('error_description', f'HTTP {response.status_code}')
                    }
                
                return result
                
        except Exception as e:
            return {'error': str(e)}
    
    def _acquire_token_by_client_credentials(self, app) -> dict:
        """Acquire token using client credentials (app-only auth)"""
        # For app-only, use .default scope
        scopes = ["https://outlook.office365.com/.default"]
        
        try:
            result = app.acquire_token_for_client(scopes=scopes)
            return result or {'error': 'No result from client credentials flow'}
        except Exception as e:
            logger.error(f"Error acquiring token by client credentials: {e}")
            return {'error': str(e)}
    
    def _load_token_cache(self):
        """Load token cache from file"""
        cache_path = Path(self.config.token_cache_file)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    self._token_cache = json.load(f)
                logger.debug(f"Loaded token cache from {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to load token cache: {e}")
    
    def _save_token_cache(self):
        """Save token cache to file"""
        cache_path = Path(self.config.token_cache_file)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(self._token_cache, f)
            logger.debug(f"Saved token cache to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save token cache: {e}")
    
    def clear_cache(self, username: Optional[str] = None):
        """Clear token cache"""
        if username:
            self._token_cache.pop(username, None)
        else:
            self._token_cache.clear()
        
        if self.config.token_cache_file:
            self._save_token_cache()


def get_initial_tokens_interactive(
    tenant_id: str,
    client_id: str,
    client_secret: Optional[str] = None
) -> dict:
    """
    Interactive OAuth2 flow to get initial refresh token.
    
    This opens a browser for user login and returns tokens.
    Run this once to get the initial refresh token, then store it securely.
    
    Args:
        tenant_id: Azure AD tenant ID
        client_id: Application client ID
        client_secret: Optional client secret
        
    Returns:
        Dict with access_token, refresh_token, etc.
    """
    try:
        import msal
    except ImportError:
        raise ImportError("MSAL library required. Install with: uv add msal")
    
    authority = f"https://login.microsoftonline.com/{tenant_id}"
    
    # Note: offline_access is implicit for public clients
    scopes = [
        "https://outlook.office365.com/IMAP.AccessAsUser.All",
        "https://outlook.office365.com/SMTP.Send",
    ]
    
    if client_secret:
        app = msal.ConfidentialClientApplication(
            client_id=client_id,
            client_credential=client_secret,
            authority=authority
        )
    else:
        app = msal.PublicClientApplication(
            client_id=client_id,
            authority=authority
        )
    
    # Interactive flow - opens browser
    result = app.acquire_token_interactive(
        scopes=scopes,
        prompt="select_account"
    )
    
    if 'error' in result:
        raise RuntimeError(f"Authentication failed: {result.get('error_description', result['error'])}")
    
    print("\n✅ Authentication successful!")
    print(f"   Account: {result.get('id_token_claims', {}).get('preferred_username', 'N/A')}")
    print(f"\n🔑 Refresh Token (store securely in .env):")
    print(f"   {result.get('refresh_token', 'N/A')[:50]}...")
    
    return result
