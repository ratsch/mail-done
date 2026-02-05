"""Database module for Phase 2"""
from .models import (
    Base, Email, EmailMetadata, SenderHistory, Classification, ReplyTracking,
    ApplicationCollection, ApplicationCollectionItem, ApplicationShareToken,
    AssignmentBatch, AssignmentBatchShare, ApplicationReviewAssignment,
)
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
    'AssignmentBatch',
    'AssignmentBatchShare',
    'ApplicationReviewAssignment',
    'get_db',
    'init_db',
    'engine',
]

