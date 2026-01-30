"""
Configuration Management

Centralized configuration using Pydantic Settings.
All settings loaded from environment variables with sensible defaults.
"""
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Ignore extra environment variables that aren't defined in the model
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields from .env that aren't in the model
    )
    
    # ============================================================
    # IMAP Configuration (optional - only needed for process_inbox.py)
    # ============================================================
    imap_host: Optional[str] = Field(None, description="IMAP server hostname")
    imap_port: int = Field(993, description="IMAP server port")
    imap_username: Optional[str] = Field(None, description="IMAP username/email")
    imap_password: Optional[str] = Field(None, description="IMAP password")
    imap_use_ssl: bool = Field(True, description="Use SSL for IMAP")
    
    # ============================================================
    # SMTP Configuration (for sending emails)
    # ============================================================
    smtp_host: Optional[str] = Field(None, description="SMTP server hostname")
    smtp_port: int = Field(587, description="SMTP server port (587=TLS, 465=SSL)")
    smtp_username: Optional[str] = Field(None, description="SMTP username (usually same as email)")
    smtp_password: Optional[str] = Field(None, description="SMTP password")
    smtp_use_tls: bool = Field(True, description="Use TLS for SMTP")
    
    # Email identity
    from_name: str = Field(
        "",
        description="Display name in From field (set via EMAIL_FROM_NAME env, defaults to empty)"
    )
    from_email: Optional[str] = Field(None, description="From email address (defaults to imap_username)")
    
    # ============================================================
    # Database Configuration
    # ============================================================
    database_url: str = Field(..., description="PostgreSQL connection URL")
    database_pool_size: int = Field(5, description="Database connection pool size")
    database_max_overflow: int = Field(10, description="Max overflow connections")
    
    # ============================================================
    # LLM Configuration
    # ============================================================
    # OpenAI
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    openai_model: str = Field("gpt-4o-mini", description="OpenAI model for classification")
    openai_embedding_model: str = Field("text-embedding-3-small", description="OpenAI embedding model")
    openai_temperature: float = Field(0.3, description="Temperature for classification (0-1)")
    
    # Anthropic (alternative)
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API key")
    anthropic_model: str = Field("claude-3-haiku-20240307", description="Anthropic model")
    
    # LLM provider preference
    llm_provider: str = Field("openai", description="LLM provider: openai or anthropic")
    
    # ============================================================
    # Reply Generation Configuration
    # ============================================================
    reply_generation_enabled: bool = Field(True, description="Enable AI reply generation")
    reply_generation_model: str = Field("gpt-4o-mini", description="Model for reply generation")
    reply_generation_temperature: float = Field(0.7, description="Temperature for reply generation")
    reply_num_variations: int = Field(2, description="Number of reply variations to generate")
    
    # ============================================================
    # Vector Search Configuration
    # ============================================================
    vector_search_enabled: bool = Field(True, description="Enable vector search")
    embedding_batch_size: int = Field(100, description="Batch size for embedding generation")
    hnsw_ef_search: int = Field(40, description="HNSW ef_search parameter (10=fast, 100=accurate)")
    similarity_threshold: float = Field(0.6, description="Default similarity threshold (0-1)")
    
    # ============================================================
    # API Configuration
    # ============================================================
    api_key: Optional[str] = Field(None, description="API key for authentication (optional for dev)")
    allowed_origins: str = Field(
        "http://localhost:3000,http://localhost:8000",
        description="Comma-separated CORS allowed origins"
    )
    api_port: int = Field(8000, description="API server port")
    
    # ============================================================
    # Processing Configuration
    # ============================================================
    process_batch_size: int = Field(100, description="Number of emails to process in batch")
    process_new_only: bool = Field(False, description="Only process unseen emails")
    dry_run: bool = Field(True, description="Dry run mode (no IMAP changes)")
    
    # ============================================================
    # IMAP Action Folders (for UI actions: spam/delete/archive)
    # ============================================================
    folder_spam: str = Field("MD/Spam", description="Folder for spam emails")
    folder_trash: str = Field("Trash", description="Folder for deleted emails")
    folder_archive: str = Field("Archive", description="Folder for archived emails")
    auto_create_folders: bool = Field(True, description="Auto-create IMAP folders if missing")
    
    # ============================================================
    # Feature Flags
    # ============================================================
    use_ai_classification: bool = Field(True, description="Enable AI classification")
    use_database: bool = Field(True, description="Enable database storage")
    use_response_tracking: bool = Field(True, description="Enable response tracking")
    use_reply_generation: bool = Field(True, description="Enable reply generation")
    use_vector_search: bool = Field(True, description="Enable vector search")
    
    # ============================================================
    # Google Drive/Sheets Configuration
    # ============================================================
    google_service_account_path: Optional[str] = Field(
        None, 
        description="Path to Google service account JSON key file"
    )
    google_drive_spreadsheet_folder_id: Optional[str] = Field(
        None,
        description="Google Drive folder ID for spreadsheets (application lists folder)"
    )
    google_drive_source_material_folder_id: Optional[str] = Field(
        None,
        description="Google Drive folder ID for application source material (attachments, email.txt, LLM responses)"
    )
    
    # ============================================================
    # Reference Letter Search Configuration
    # ============================================================
    reference_letter_exclude_senders: str = Field(
        "",
        description="Comma-separated list of your email addresses to exclude from reference letter search (set via REFERENCE_LETTER_EXCLUDE_SENDERS env)"
    )
    
    # ============================================================
    # Logging Configuration
    # ============================================================
    log_level: str = Field("INFO", description="Logging level (DEBUG/INFO/WARNING/ERROR)")
    log_format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    
    @property
    def allowed_origins_list(self) -> List[str]:
        """Parse allowed origins into list."""
        return [origin.strip() for origin in self.allowed_origins.split(",")]
    
    @property
    def reference_letter_exclude_senders_list(self) -> List[str]:
        """Parse excluded sender addresses into list."""
        if not self.reference_letter_exclude_senders:
            return []
        return [addr.strip().lower() for addr in self.reference_letter_exclude_senders.split(",") if addr.strip()]
    
    @property
    def from_email_address(self) -> str:
        """Get from email (defaults to IMAP username)."""
        return self.from_email or self.imap_username


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get application settings (singleton).
    
    Returns:
        Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings():
    """Reload settings from environment (useful for testing)."""
    global _settings
    _settings = Settings()
    return _settings

