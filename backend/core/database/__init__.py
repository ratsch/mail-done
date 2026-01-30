"""Database module for Phase 2"""
from .models import Base, Email, EmailMetadata, SenderHistory, Classification, ReplyTracking, ApplicationCollection, ApplicationCollectionItem
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
    'get_db',
    'init_db',
    'engine',
]

