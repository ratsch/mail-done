"""
Authentication providers for email accounts.

Supports multiple authentication methods:
- Password: Traditional username/password IMAP auth
- OAuth2: Microsoft 365/Outlook OAuth2 with XOAUTH2
"""

from .oauth2_provider import OAuth2Provider, OAuth2Config

__all__ = ['OAuth2Provider', 'OAuth2Config']
