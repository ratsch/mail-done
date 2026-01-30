"""
Response Tracker

Manages reply tracking in database, identifies unanswered emails,
and monitors response status.
"""
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from backend.core.database.models import Email, EmailMetadata, ReplyTracking, SenderHistory
from backend.core.tracking.reply_detector import ReplyDetector, ReplyAnalysis


class ResponseTracker:
    """
    Tracks which emails need replies and monitors response status.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.detector = ReplyDetector()
    
    async def analyze_and_track(
        self,
        email: Email,
        ai_metadata: dict,
        vip_level: Optional[str] = None
    ) -> Optional[ReplyTracking]:
        """
        Analyze if email needs reply and create tracking record.
        
        Args:
            email: Email object
            ai_metadata: AI classification metadata
            vip_level: VIP level if applicable
        
        Returns:
            ReplyTracking object if reply needed, None otherwise
        """
        # Get AI category
        category = ai_metadata.get('ai_category')
        
        # Analyze reply need
        analysis = self.detector.analyze(
            subject=email.subject,
            body=email.body_markdown or email.body_text or '',
            ai_metadata=ai_metadata,
            vip_level=vip_level,
            category=category
        )
        
        # If reply not needed, return None
        if not analysis.needs_reply:
            return None
        
        # Check if tracking already exists
        existing = self.db.query(ReplyTracking).filter(
            ReplyTracking.email_id == email.id
        ).first()
        
        if existing:
            # Update existing tracking
            existing.needs_reply = True
            existing.reply_deadline = analysis.deadline
            existing.detected_by = analysis.detected_by
            existing.question_detected = analysis.question_detected
            existing.action_requested = analysis.action_requested
            existing.updated_at = datetime.utcnow()
            tracking = existing
        else:
            # Create new tracking
            tracking = ReplyTracking(
                email_id=email.id,
                needs_reply=True,
                reply_deadline=analysis.deadline,
                detected_by=analysis.detected_by,
                question_detected=analysis.question_detected,
                action_requested=analysis.action_requested
            )
            self.db.add(tracking)
        
        # Also update email_metadata
        if email.email_metadata:
            email.email_metadata.needs_reply = True
            email.email_metadata.reply_deadline = analysis.deadline
            # Store priority in user_priority or priority_score
            email.email_metadata.priority_score = analysis.priority * 10  # Scale to 0-100
        
        # Don't commit here - let caller handle transaction
        # self.db.commit()  # Removed to avoid transaction conflicts
        self.db.flush()  # Flush changes, commit happens in caller
        return tracking
    
    def get_unanswered_emails(
        self,
        min_priority: int = 7,
        vip_only: bool = False,
        categories: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Tuple[Email, int]]:
        """
        Get all unanswered important emails.
        
        Args:
            min_priority: Minimum priority score (1-10)
            vip_only: Only include VIP emails
            categories: Filter by categories
            limit: Maximum results
        
        Returns:
            List of (Email, priority_score) tuples sorted by priority
        """
        # Base query: emails with tracking that need reply and haven't been replied
        query = self.db.query(Email, EmailMetadata).join(
            EmailMetadata, Email.id == EmailMetadata.email_id
        ).join(
            ReplyTracking, Email.id == ReplyTracking.email_id
        ).filter(
            ReplyTracking.needs_reply == True,
            ReplyTracking.replied_at.is_(None),
            Email.is_answered == False  # Also check IMAP flag
        )
        
        # Apply VIP filter
        if vip_only:
            query = query.filter(EmailMetadata.vip_level.in_(['urgent', 'high', 'medium']))
        
        # Apply category filter
        if categories:
            query = query.filter(EmailMetadata.ai_category.in_(categories))
        
        # Apply priority filter
        min_priority_score = min_priority * 10  # Convert to 0-100 scale
        query = query.filter(
            or_(
                EmailMetadata.priority_score >= min_priority_score,
                EmailMetadata.vip_level.in_(['urgent', 'high'])  # Always include high VIPs
            )
        )
        
        # Sort by priority (deadline first, then VIP, then urgency, then date)
        query = query.order_by(
            # Overdue deadlines first
            ReplyTracking.reply_deadline.asc().nullslast(),
            # Then VIP level
            EmailMetadata.vip_level.desc().nullslast(),
            # Then priority score
            EmailMetadata.priority_score.desc().nullslast(),
            # Finally by date (newest first)
            Email.date.desc()
        )
        
        # Limit results
        results = query.limit(limit).all()
        
        # Return as (Email, priority) tuples
        return [
            (email, metadata.priority_score or 50) 
            for email, metadata in results
        ]
    
    def get_overdue_replies(self, max_age_hours: int = 72) -> List[Tuple[Email, timedelta]]:
        """
        Get emails overdue for reply.
        
        Args:
            max_age_hours: Maximum hours without reply for non-deadline emails
                          (VIPs: 24h, normal: 72h)
        
        Returns:
            List of (Email, time_overdue) tuples
        """
        now = datetime.utcnow()
        
        # Emails with explicit deadlines that passed
        deadline_overdue = self.db.query(Email, ReplyTracking).join(
            ReplyTracking, Email.id == ReplyTracking.email_id
        ).filter(
            ReplyTracking.needs_reply == True,
            ReplyTracking.replied_at.is_(None),
            ReplyTracking.reply_deadline.isnot(None),
            ReplyTracking.reply_deadline < now
        ).all()
        
        # Emails without deadlines but VIP and old
        vip_overdue = self.db.query(Email, EmailMetadata).join(
            EmailMetadata, Email.id == EmailMetadata.email_id
        ).join(
            ReplyTracking, Email.id == ReplyTracking.email_id
        ).filter(
            ReplyTracking.needs_reply == True,
            ReplyTracking.replied_at.is_(None),
            ReplyTracking.reply_deadline.is_(None),  # No explicit deadline
            EmailMetadata.vip_level.in_(['urgent', 'high']),
            Email.date < now - timedelta(hours=24)  # VIPs: 24h
        ).all()
        
        # Normal emails without deadlines but very old
        normal_overdue = self.db.query(Email, EmailMetadata).join(
            EmailMetadata, Email.id == EmailMetadata.email_id
        ).join(
            ReplyTracking, Email.id == ReplyTracking.email_id
        ).filter(
            ReplyTracking.needs_reply == True,
            ReplyTracking.replied_at.is_(None),
            ReplyTracking.reply_deadline.is_(None),  # No explicit deadline
            or_(
                EmailMetadata.vip_level.is_(None),
                EmailMetadata.vip_level == 'medium'
            ),
            Email.date < now - timedelta(hours=max_age_hours)
        ).all()
        
        # Combine and calculate overdue time
        overdue = []
        
        for email, tracking in deadline_overdue:
            time_overdue = now - tracking.reply_deadline
            overdue.append((email, time_overdue))
        
        for email, metadata in vip_overdue:
            time_overdue = now - email.date - timedelta(hours=24)
            overdue.append((email, time_overdue))
        
        for email, metadata in normal_overdue:
            time_overdue = now - email.date - timedelta(hours=max_age_hours)
            overdue.append((email, time_overdue))
        
        # Sort by overdue time (most overdue first)
        overdue.sort(key=lambda x: x[1], reverse=True)
        
        return overdue
    
    def mark_replied(
        self,
        email_id: UUID,
        reply_message_id: Optional[str] = None
    ) -> bool:
        """
        Mark email as replied.
        
        Args:
            email_id: Email that was replied to
            reply_message_id: Optional Message-ID of your reply
        
        Returns:
            True if successful
        """
        import logging
        logger = logging.getLogger(__name__)
        
        tracking = self.db.query(ReplyTracking).filter(
            ReplyTracking.email_id == email_id
        ).first()
        
        if not tracking:
            logger.warning(f"No tracking found for email {email_id}")
            return False
        
        try:
            # Start transaction
            # Update tracking
            tracking.replied_at = datetime.utcnow()
            
            if reply_message_id:
                # Try to find the reply email in database
                reply_email = self.db.query(Email).filter(
                    Email.message_id == reply_message_id
                ).first()
                if reply_email:
                    tracking.reply_email_id = reply_email.id
                    logger.info(f"Linked reply {reply_message_id} to email {email_id}")
                else:
                    logger.warning(
                        f"Reply message {reply_message_id} not found in database. "
                        f"Reply tracking will still be marked complete, but reply won't be linked."
                    )
            
            tracking.needs_reply = False
            tracking.updated_at = datetime.utcnow()
            
            # Update email metadata
            email = self.db.query(Email).filter(Email.id == email_id).first()
            if not email:
                logger.error(f"Email {email_id} not found")
                self.db.rollback()
                return False
            
            if email.email_metadata:
                email.email_metadata.replied_at = datetime.utcnow()
                email.email_metadata.needs_reply = False
            
            # Update email flags
            email.is_answered = True
            
            # Update sender history (track response)
            sender = self.db.query(SenderHistory).filter(
                SenderHistory.email_address == email.from_address
            ).first()
            if sender:
                sender.reply_count += 1
                # Update average reply time (exponential moving average)
                reply_time_hours = (datetime.utcnow() - email.date).total_seconds() / 3600
                if sender.avg_reply_time_hours:
                    # EMA with alpha=0.2 (standard approach)
                    sender.avg_reply_time_hours = (
                        sender.avg_reply_time_hours * 0.8 + reply_time_hours * 0.2
                    )
                else:
                    sender.avg_reply_time_hours = reply_time_hours
                
                sender.updated_at = datetime.utcnow()
            
            # Commit all changes atomically
            self.db.commit()
            logger.info(f"Successfully marked email {email_id} as replied")
            return True
            
        except Exception as e:
            # Rollback on any error
            logger.error(f"Error marking email {email_id} as replied: {e}")
            self.db.rollback()
            return False
    
    def snooze_reply(
        self,
        email_id: UUID,
        until: datetime
    ) -> bool:
        """
        Snooze reply reminder until specified time.
        
        Args:
            email_id: Email to snooze
            until: When to remind again
        
        Returns:
            True if successful
        """
        tracking = self.db.query(ReplyTracking).filter(
            ReplyTracking.email_id == email_id
        ).first()
        
        if not tracking:
            return False
        
        # Store in reminded_at (we'll use this as "snoozed_until")
        tracking.reminded_at = until
        tracking.updated_at = datetime.utcnow()
        
        self.db.commit()
        return True
    
    def get_reply_stats(self) -> dict:
        """
        Get reply tracking statistics.
        
        Returns:
            Dictionary with stats:
            - total_needing_reply
            - overdue_count
            - by_category
            - by_vip_level
            - avg_response_time_hours
        """
        # Total needing reply
        total_needing_reply = self.db.query(ReplyTracking).filter(
            ReplyTracking.needs_reply == True,
            ReplyTracking.replied_at.is_(None)
        ).count()
        
        # Overdue count
        now = datetime.utcnow()
        overdue_count = self.db.query(ReplyTracking).filter(
            ReplyTracking.needs_reply == True,
            ReplyTracking.replied_at.is_(None),
            ReplyTracking.reply_deadline < now
        ).count()
        
        # By category
        by_category = {}
        category_results = self.db.query(
            EmailMetadata.ai_category,
            self.db.func.count(ReplyTracking.id)
        ).join(
            ReplyTracking, EmailMetadata.email_id == ReplyTracking.email_id
        ).filter(
            ReplyTracking.needs_reply == True,
            ReplyTracking.replied_at.is_(None)
        ).group_by(EmailMetadata.ai_category).all()
        
        for category, count in category_results:
            by_category[category or 'unknown'] = count
        
        # By VIP level
        by_vip = {}
        vip_results = self.db.query(
            EmailMetadata.vip_level,
            self.db.func.count(ReplyTracking.id)
        ).join(
            ReplyTracking, EmailMetadata.email_id == ReplyTracking.email_id
        ).filter(
            ReplyTracking.needs_reply == True,
            ReplyTracking.replied_at.is_(None)
        ).group_by(EmailMetadata.vip_level).all()
        
        for vip_level, count in vip_results:
            by_vip[vip_level or 'none'] = count
        
        # Average response time (from sender_history)
        avg_response_time = self.db.query(
            self.db.func.avg(SenderHistory.avg_reply_time_hours)
        ).scalar()
        
        return {
            'total_needing_reply': total_needing_reply,
            'overdue_count': overdue_count,
            'by_category': by_category,
            'by_vip_level': by_vip,
            'avg_response_time_hours': round(avg_response_time, 1) if avg_response_time else None
        }

