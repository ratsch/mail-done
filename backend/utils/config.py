"""
Configuration management using environment variables.
Follows inbox-zero's configuration patterns.
"""
import os
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
import json
import logging

logger = logging.getLogger(__name__)


class IMAPAccountConfig(BaseSettings):
    """Single IMAP account configuration"""
    id: str
    host: str
    username: str
    password: str
    port: int = 993
    use_ssl: bool = True
    folder: str = 'INBOX'


class SMTPAccountConfig(BaseSettings):
    """SMTP configuration for sending emails"""
    host: str
    username: str
    password: str
    port: int = 587  # STARTTLS default
    use_tls: bool = True
    use_ssl: bool = False
    from_address: Optional[str] = None


class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # IMAP Configuration (primary account for simple setup)
    imap_host: str = Field(default="imap.gmail.com", env="IMAP_HOST")
    imap_port: int = Field(default=993, env="IMAP_PORT")
    imap_username: str = Field(default="", env="IMAP_USERNAME")
    imap_password: str = Field(default="", env="IMAP_PASSWORD")
    
    # SMTP Configuration (for sending emails/replies)
    smtp_host: str = Field(default="smtp.gmail.com", env="SMTP_HOST")
    smtp_port: int = Field(default=587, env="SMTP_PORT")
    smtp_username: str = Field(default="", env="SMTP_USERNAME")
    smtp_password: str = Field(default="", env="SMTP_PASSWORD")
    smtp_use_tls: bool = Field(default=True, env="SMTP_USE_TLS")
    smtp_from_address: Optional[str] = Field(default=None, env="SMTP_FROM_ADDRESS")
    
    # Multi-account support (JSON array in env var, optional)
    # Example: IMAP_ACCOUNTS='[{"id":"work","host":"imap.gmail.com","username":"...","password":"..."}]'
    imap_accounts_json: Optional[str] = Field(default=None, env="IMAP_ACCOUNTS")
    
    @property
    def imap_accounts(self) -> List[IMAPAccountConfig]:
        """Get list of IMAP accounts (primary + additional)"""
        accounts = []
        
        # Add primary account if configured
        if self.imap_username and self.imap_password:
            accounts.append(IMAPAccountConfig(
                id="primary",
                host=self.imap_host,
                port=self.imap_port,
                username=self.imap_username,
                password=self.imap_password,
            ))
        
        # Add additional accounts from JSON
        if self.imap_accounts_json:
            try:
                additional = json.loads(self.imap_accounts_json)
                for acc in additional:
                    accounts.append(IMAPAccountConfig(**acc))
            except Exception as e:
                logger.warning(f"Failed to parse IMAP_ACCOUNTS: {e}")
        
        return accounts
    
    # LLM Configuration
    llm_provider: str = Field(default="openai", env="LLM_PROVIDER")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-3.5-turbo", env="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.3, env="OPENAI_TEMPERATURE")
    
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-haiku-20240307", env="ANTHROPIC_MODEL")
    
    # Database
    database_url: str = Field(default="sqlite:///./email.db", env="DATABASE_URL")
    
    # Safety
    dry_run: bool = Field(default=True, env="DRY_RUN")
    
    # Application
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    max_emails_per_run: int = Field(default=10, env="MAX_EMAILS_PER_RUN")
    imap_debug: bool = Field(default=False, env="IMAP_DEBUG")
    imap_timeout: int = Field(default=120, env="IMAP_TIMEOUT")  # Network timeout in seconds (increased from 30)
    
    # Folders
    folder_receipts: str = Field(default="Receipts", env="FOLDER_RECEIPTS")
    folder_newsletters: str = Field(default="Newsletters", env="FOLDER_NEWSLETTERS")
    folder_archive: str = Field(default="Archive", env="FOLDER_ARCHIVE")
    folder_urgent: str = Field(default="Urgent", env="FOLDER_URGENT")
    
    # Colors
    urgent_color: int = Field(default=1, env="URGENT_COLOR")  # Red
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get settings singleton"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings():
    """Reload settings from environment"""
    global _settings
    _settings = Settings()
    return _settings

