"""Database module for Phase 2"""
from .models import Base, Email, EmailMetadata, SenderHistory, Classification, ReplyTracking, ApplicationCollection, ApplicationCollectionItem, ApplicationShareToken
from .connection import get_db, init_db, engine

__all__ = [
    'Base',
    'Email',
    'EmailMetadata',
    'SenderHistory',
    'Classification',
    'ReplyTracking',
    'ApplicationCollection',
    'ApplicationCollectionItem',
    'ApplicationShareToken',
    'get_db',
    'init_db',
    'engine',
]

