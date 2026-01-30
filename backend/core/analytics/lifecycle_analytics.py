"""
Email Lifecycle Analytics

Analyze email folder movements and user behavior patterns.
Provides insights into:
- Response time patterns
- Archive vs Delete decisions
- Sender importance signals
- Rule accuracy
"""
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, desc

from backend.core.database.models import Email, EmailMetadata, EmailLocationHistory, SenderHistory


class LifecycleAnalytics:
    """Analytics for email lifecycle and user behavior."""
    
    def __init__(self, db: Session):
        """
        Initialize analytics.
        
        Args:
            db: Database session
        """
        self.db = db
    
    def get_response_time_by_sender(self, limit: int = 50) -> List[Dict]:
        """
        Get average response time for each sender.
        Shows who you respond to quickly (important) vs slowly.
        
        Args:
            limit: Max number of senders to return
            
        Returns:
            List of {sender, avg_seconds, count} sorted by fastest response
        """
        results = self.db.query(
            Email.from_address,
            func.avg(EmailMetadata.time_to_action_seconds).label('avg_time'),
            func.count(Email.id).label('count')
        ).join(
            EmailMetadata, Email.id == EmailMetadata.email_id
        ).filter(
            EmailMetadata.time_to_action_seconds.isnot(None)
        ).group_by(
            Email.from_address
        ).having(
            func.count(Email.id) >= 3  # At least 3 emails
        ).order_by(
            func.avg(EmailMetadata.time_to_action_seconds)
        ).limit(limit).all()
        
        return [
            {
                'sender': sender,
                'avg_response_seconds': int(avg_time),
                'avg_response_hours': round(avg_time / 3600, 1),
                'email_count': count
            }
            for sender, avg_time, count in results
        ]
    
    def get_archive_vs_delete_patterns(self) -> Dict:
        """
        Analyze what types of emails you archive vs delete.
        
        Returns:
            Dict with archive/delete statistics
        """
        total_archived = self.db.query(func.count(EmailMetadata.id)).filter(
            EmailMetadata.user_archived == True
        ).scalar() or 0
        
        total_deleted = self.db.query(func.count(EmailMetadata.id)).filter(
            EmailMetadata.user_deleted == True
        ).scalar() or 0
        
        # Archive by category
        archived_by_category = self.db.query(
            EmailMetadata.ai_category,
            func.count(EmailMetadata.id).label('count')
        ).filter(
            EmailMetadata.user_archived == True,
            EmailMetadata.ai_category.isnot(None)
        ).group_by(
            EmailMetadata.ai_category
        ).order_by(
            desc('count')
        ).limit(10).all()
        
        # Delete by category
        deleted_by_category = self.db.query(
            EmailMetadata.ai_category,
            func.count(EmailMetadata.id).label('count')
        ).filter(
            EmailMetadata.user_deleted == True,
            EmailMetadata.ai_category.isnot(None)
        ).group_by(
            EmailMetadata.ai_category
        ).order_by(
            desc('count')
        ).limit(10).all()
        
        return {
            'total_archived': total_archived,
            'total_deleted': total_deleted,
            'archived_by_category': [
                {'category': cat, 'count': count}
                for cat, count in archived_by_category
            ],
            'deleted_by_category': [
                {'category': cat, 'count': count}
                for cat, count in deleted_by_category
            ]
        }
    
    def get_rule_accuracy(self) -> List[Dict]:
        """
        Check rule accuracy by finding emails that were auto-filed but user later moved.
        
        Returns:
            List of moves that suggest rule corrections needed
        """
        # Find emails that were auto-filed, then user moved them
        user_corrections = self.db.query(EmailLocationHistory).filter(
            EmailLocationHistory.moved_by == 'user',
            EmailLocationHistory.email_id.in_(
                self.db.query(EmailMetadata.email_id).filter(
                    EmailMetadata.auto_filed == True
                )
            )
        ).limit(100).all()
        
        corrections = []
        for correction in user_corrections:
            # Get the auto-file history for this email
            auto_move = self.db.query(EmailLocationHistory).filter(
                EmailLocationHistory.email_id == correction.email_id,
                EmailLocationHistory.moved_by.in_(['rule', 'ai']),
                EmailLocationHistory.moved_at < correction.moved_at
            ).order_by(EmailLocationHistory.moved_at.desc()).first()
            
            if auto_move:
                email = self.db.query(Email).filter(Email.id == correction.email_id).first()
                corrections.append({
                    'email_id': str(correction.email_id),
                    'subject': email.subject if email else 'Unknown',
                    'auto_filed_to': auto_move.to_folder,
                    'auto_filed_by': auto_move.moved_by,
                    'auto_filed_reason': auto_move.move_reason,
                    'user_moved_to': correction.to_folder,
                    'time_before_correction_hours': round(
                        (correction.moved_at - auto_move.moved_at).total_seconds() / 3600, 1
                    )
                })
        
        return corrections
    
    def get_folder_statistics(self, days: int = 30) -> List[Dict]:
        """
        Get folder activity statistics.
        
        Args:
            days: Look back this many days
            
        Returns:
            List of folder stats with movement counts
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        stats = self.db.query(
            EmailLocationHistory.to_folder,
            func.count(EmailLocationHistory.id).label('move_count'),
            func.count(func.distinct(EmailLocationHistory.email_id)).label('unique_emails')
        ).filter(
            EmailLocationHistory.moved_at >= cutoff
        ).group_by(
            EmailLocationHistory.to_folder
        ).order_by(
            desc('move_count')
        ).all()
        
        return [
            {
                'folder': folder,
                'moves': move_count,
                'unique_emails': unique
            }
            for folder, move_count, unique in stats
        ]
    
    def get_importance_distribution(self) -> Dict:
        """
        Get distribution of importance scores.
        
        Returns:
            Dict with importance score statistics
        """
        # Count by importance score
        distribution = self.db.query(
            EmailMetadata.importance_score,
            func.count(EmailMetadata.id).label('count')
        ).filter(
            EmailMetadata.importance_score.isnot(None)
        ).group_by(
            EmailMetadata.importance_score
        ).order_by(
            EmailMetadata.importance_score
        ).all()
        
        total = sum(count for _, count in distribution)
        
        return {
            'distribution': [
                {
                    'score': score,
                    'count': count,
                    'percentage': round(count / total * 100, 1) if total > 0 else 0
                }
                for score, count in distribution
            ],
            'total_scored': total,
            'avg_importance': self.db.query(
                func.avg(EmailMetadata.importance_score)
            ).filter(
                EmailMetadata.importance_score.isnot(None)
            ).scalar() or 0
        }
    
    def get_quick_vs_slow_responders(self, threshold_hours: int = 24) -> Dict:
        """
        Categorize senders by response time.
        
        Args:
            threshold_hours: Cutoff for "quick" response
            
        Returns:
            Dict with quick vs slow responder lists
        """
        threshold_seconds = threshold_hours * 3600
        
        # Quick responders (you act within threshold)
        quick = self.db.query(
            Email.from_address,
            func.avg(EmailMetadata.time_to_action_seconds).label('avg_time'),
            func.count(Email.id).label('count')
        ).join(
            EmailMetadata, Email.id == EmailMetadata.email_id
        ).filter(
            EmailMetadata.time_to_action_seconds.isnot(None),
            EmailMetadata.time_to_action_seconds < threshold_seconds
        ).group_by(
            Email.from_address
        ).having(
            func.count(Email.id) >= 2
        ).order_by(
            func.avg(EmailMetadata.time_to_action_seconds)
        ).limit(20).all()
        
        # Slow responders (you take longer)
        slow = self.db.query(
            Email.from_address,
            func.avg(EmailMetadata.time_to_action_seconds).label('avg_time'),
            func.count(Email.id).label('count')
        ).join(
            EmailMetadata, Email.id == EmailMetadata.email_id
        ).filter(
            EmailMetadata.time_to_action_seconds.isnot(None),
            EmailMetadata.time_to_action_seconds >= threshold_seconds
        ).group_by(
            Email.from_address
        ).having(
            func.count(Email.id) >= 2
        ).order_by(
            desc(func.avg(EmailMetadata.time_to_action_seconds))
        ).limit(20).all()
        
        return {
            'threshold_hours': threshold_hours,
            'quick_responders': [
                {
                    'sender': sender,
                    'avg_hours': round(avg / 3600, 1),
                    'count': count
                }
                for sender, avg, count in quick
            ],
            'slow_responders': [
                {
                    'sender': sender,
                    'avg_hours': round(avg / 3600, 1),
                    'count': count
                }
                for sender, avg, count in slow
            ]
        }
    
    def get_email_journey(self, email_id: str) -> Dict:
        """
        Get complete journey of a specific email through folders.
        
        Args:
            email_id: UUID of email
            
        Returns:
            Dict with email info and movement timeline
        """
        from uuid import UUID
        
        email = self.db.query(Email).filter(Email.id == UUID(email_id)).first()
        if not email:
            return {'error': 'Email not found'}
        
        metadata = self.db.query(EmailMetadata).filter(
            EmailMetadata.email_id == email.id
        ).first()
        
        history = self.db.query(EmailLocationHistory).filter(
            EmailLocationHistory.email_id == email.id
        ).order_by(EmailLocationHistory.moved_at).all()
        
        return {
            'email': {
                'message_id': email.message_id,
                'subject': email.subject,
                'from': email.from_address,
                'date': email.date.isoformat() if email.date else None,
                'current_folder': email.folder
            },
            'lifecycle': {
                'status': metadata.email_status if metadata else None,
                'first_seen_folder': metadata.first_seen_folder if metadata else None,
                'first_seen_at': metadata.first_seen_at.isoformat() if metadata and metadata.first_seen_at else None,
                'move_count': metadata.folder_move_count if metadata else 0,
                'user_archived': metadata.user_archived if metadata else False,
                'user_deleted': metadata.user_deleted if metadata else False,
                'importance_score': metadata.importance_score if metadata else None,
                'time_to_action_hours': round(metadata.time_to_action_seconds / 3600, 1) if metadata and metadata.time_to_action_seconds else None
            },
            'journey': [
                {
                    'from_folder': h.from_folder,
                    'to_folder': h.to_folder,
                    'moved_at': h.moved_at.isoformat(),
                    'moved_by': h.moved_by,
                    'reason': h.move_reason,
                    'time_in_previous_hours': round(h.time_in_previous_folder_seconds / 3600, 1) if h.time_in_previous_folder_seconds else None
                }
                for h in history
            ]
        }

