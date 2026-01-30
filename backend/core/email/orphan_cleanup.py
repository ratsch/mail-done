"""
Orphan Cleanup Service

Periodic service to detect and clean up orphaned/duplicated emails after cross-account moves.
Handles:
- Duplicate detection (same Message-ID in multiple accounts)
- Retry failed moves
- Clean up orphaned emails (exist in IMAP but DB thinks moved)
"""
from typing import Dict
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func
import logging

from backend.core.database.models import Email, CrossAccountMove
from backend.core.email.imap_monitor import IMAPMonitor
from backend.core.email.cross_account_move import CrossAccountMoveService
from backend.core.accounts.manager import AccountManager

logger = logging.getLogger(__name__)


class OrphanCleanupService:
    """
    Periodic service to detect and clean up orphaned/duplicated emails after cross-account moves.
    
    Runs three cleanup operations:
    1. Detect and remove duplicates (same Message-ID in multiple accounts)
    2. Retry failed moves (check next_retry_at)
    3. Clean up orphaned emails (exist in IMAP but DB thinks moved)
    """
    
    def __init__(self, 
                 db_session: Session, 
                 account_manager: AccountManager, 
                 dry_run: bool = False):
        """
        Initialize orphan cleanup service.
        
        Args:
            db_session: Database session
            account_manager: AccountManager instance
            dry_run: If True, only log what would happen
        """
        self.db = db_session
        self.account_manager = account_manager
        self.dry_run = dry_run
        self.stats = {
            'duplicates_detected': 0,
            'duplicates_removed': 0,
            'failed_moves_retried': 0,
            'orphans_cleaned': 0
        }
    
    async def run_cleanup(self) -> Dict:
        """
        Run full cleanup cycle.
        
        Returns:
            Dictionary with cleanup statistics
        """
        logger.info("Starting orphan cleanup cycle...")
        
        # 1. Detect and remove duplicates
        await self._cleanup_duplicates()
        
        # 2. Retry failed moves
        await self._retry_failed_moves()
        
        # 3. Clean up orphaned emails (exist in IMAP but DB thinks moved)
        await self._cleanup_orphans()
        
        logger.info(f"Cleanup complete: {self.stats}")
        return self.stats
    
    async def _cleanup_duplicates(self):
        """Find and remove duplicate emails across accounts"""
        logger.info("Checking for duplicate emails...")
        
        # Find emails with same Message-ID in multiple accounts
        duplicates_query = (
            self.db.query(Email.message_id)
            .group_by(Email.message_id)
            .having(func.count(Email.id) > 1)
            .all()
        )
        
        for (message_id,) in duplicates_query:
            if not message_id:  # Skip emails without Message-ID
                continue
                
            emails = self.db.query(Email).filter(
                Email.message_id == message_id
            ).all()
            
            if len(emails) < 2:
                continue
            
            self.stats['duplicates_detected'] += 1
            logger.warning(f"Duplicate detected: {message_id} in accounts: "
                          f"{[e.account_id for e in emails]}")
            
            # Check if there's a completed cross-account move
            move = self.db.query(CrossAccountMove).filter(
                CrossAccountMove.message_id == message_id,
                CrossAccountMove.status == 'completed'
            ).order_by(CrossAccountMove.completed_at.desc()).first()
            
            if move:
                # Keep the email in target account, delete from source
                for email in emails:
                    if email.account_id == move.from_account_id:
                        logger.info(f"Removing duplicate from source account: {email.account_id}")
                        
                        if not self.dry_run:
                            # Delete from IMAP
                            await self._delete_from_imap(email)
                            # Delete from DB
                            self.db.delete(email)
                            self.stats['duplicates_removed'] += 1
                
                # Mark move as resolved
                move.duplicate_detected = True
                move.resolved_at = datetime.utcnow()
        
        if not self.dry_run:
            self.db.commit()
    
    async def _retry_failed_moves(self):
        """Retry failed cross-account moves that are scheduled for retry"""
        logger.info("Checking for failed moves to retry...")
        
        # Find failed moves that haven't exceeded retry limit
        failed_moves = self.db.query(CrossAccountMove).filter(
            CrossAccountMove.status.in_(['failed', 'retrying']),
            CrossAccountMove.retry_count < CrossAccountMove.max_retries,
            CrossAccountMove.next_retry_at <= datetime.utcnow()
        ).all()
        
        for move in failed_moves:
            logger.info(f"Retrying failed move: {move.message_id} "
                       f"(attempt {move.retry_count + 1}/{move.max_retries})")
            
            if self.dry_run:
                logger.info(f"[DRY RUN] Would retry move {move.id}")
                continue
            
            # Get email from DB
            email = self.db.query(Email).filter(Email.id == move.email_id).first()
            if not email:
                logger.error(f"Email not found for move {move.id} (email_id: {move.email_id})")
                move.status = 'failed'
                move.error_message = "Email not found in database"
                self.db.commit()  # Commit the failed status
                continue
            
            # Update move status
            move.status = 'retrying'
            move.retry_count += 1
            move.last_retry_at = datetime.utcnow()
            
            # Calculate next retry time (exponential backoff)
            retry_manager = CrossAccountMoveService(
                self.account_manager,
                dry_run=self.dry_run,
                max_retries=move.max_retries
            ).retry_manager
            move.next_retry_at = retry_manager.calculate_next_retry(move.retry_count)
            
            # Retry the move
            try:
                # Get source IMAP connection
                from_config = self.account_manager.get_imap_config(move.from_account_id)
                from_imap = IMAPMonitor(from_config)
                from_imap.connect()
                
                # Handle UIDVALIDITY changes: if email.uid is missing or invalid,
                # look up the message_id to find the current UID
                email_uid = str(email.uid) if email.uid else None
                
                if not email_uid:
                    logger.warning(f"Email {email.id} has no UID, looking up by Message-ID")
                    from_imap.client.select_folder(move.from_folder)
                    # Search for message by Message-ID header
                    search_results = from_imap.client.search(['HEADER', 'Message-ID', move.message_id])
                    if search_results:
                        email_uid = str(search_results[0])  # Use first match
                        logger.info(f"Found email with new UID: {email_uid}")
                    else:
                        logger.error(f"Email {move.message_id} not found in {move.from_account_id}:{move.from_folder}")
                        move.status = 'failed'
                        move.error_message = "Email not found in source folder (may have been moved/deleted)"
                        from_imap.disconnect()
                        self.db.commit()
                        continue
                
                # Create cross-account service and retry
                cross_account_service = CrossAccountMoveService(
                    self.account_manager,
                    dry_run=self.dry_run,
                    max_retries=move.max_retries - move.retry_count  # Remaining retries
                )
                
                success, error, move_method = await cross_account_service.move_email(
                    email_uid=email_uid,
                    message_id=move.message_id,
                    from_account=move.from_account_id,
                    from_folder=move.from_folder,
                    to_account=move.to_account_id,
                    to_folder=move.to_folder,
                    from_imap=from_imap
                )
                
                from_imap.disconnect()
                
                if success:
                    move.status = 'completed'
                    move.completed_at = datetime.utcnow()
                    move.move_method = move_method or 'unknown'
                    move.error_message = None
                    self.stats['failed_moves_retried'] += 1
                    logger.info(f"Successfully retried move {move.id}")
                else:
                    if move.retry_count >= move.max_retries:
                        move.status = 'failed'
                        logger.error(f"Move {move.id} failed after {move.retry_count} retries")
                    else:
                        move.status = 'retrying'
                    move.error_message = error
                    
            except Exception as e:
                logger.error(f"Error retrying move {move.id}: {e}")
                move.status = 'failed'
                move.error_message = str(e)
        
        if not self.dry_run:
            self.db.commit()
    
    async def _cleanup_orphans(self):
        """Clean up emails that exist in IMAP but DB thinks they've moved"""
        logger.info("Checking for orphaned emails...")
        
        # Find completed moves from last 7 days
        recent_moves = self.db.query(CrossAccountMove).filter(
            CrossAccountMove.status == 'completed',
            CrossAccountMove.completed_at > datetime.utcnow() - timedelta(days=7)
        ).all()
        
        for move in recent_moves:
            # Check if email still exists in source account IMAP
            source_config = self.account_manager.get_imap_config(move.from_account_id)
            source_imap = IMAPMonitor(source_config)
            
            try:
                source_imap.connect()
                source_imap.client.select_folder(move.from_folder)
                
                # Search for email by Message-ID
                matches = source_imap.client.search(['HEADER', 'Message-ID', move.message_id])
                
                if matches:
                    logger.warning(f"Orphan detected: {move.message_id} still in "
                                 f"{move.from_account_id}:{move.from_folder} after completed move")
                    self.stats['orphans_cleaned'] += 1
                    
                    if not self.dry_run:
                        # Delete the orphan
                        for uid in matches:
                            source_imap.client.set_flags([uid], [b'\\Deleted'])
                        source_imap.client.expunge()
                        logger.info(f"Deleted orphan email {move.message_id} from source")
                        
            except Exception as e:
                logger.error(f"Error checking for orphan {move.message_id}: {e}")
            finally:
                source_imap.disconnect()
    
    async def _delete_from_imap(self, email: Email):
        """
        Delete email from IMAP server.
        
        Args:
            email: Email model instance
        """
        try:
            imap_config = self.account_manager.get_imap_config(email.account_id)
            imap = IMAPMonitor(imap_config)
            imap.connect()
            imap.client.select_folder(email.folder)
            
            # Find by Message-ID
            matches = imap.client.search(['HEADER', 'Message-ID', email.message_id])
            if matches:
                for uid in matches:
                    imap.client.set_flags([uid], [b'\\Deleted'])
                imap.client.expunge()
                logger.info(f"Deleted email {email.message_id} from {email.account_id}:{email.folder}")
            else:
                logger.warning(f"Email {email.message_id} not found in IMAP {email.account_id}:{email.folder}")
                
            imap.disconnect()
        except Exception as e:
            logger.error(f"Failed to delete email {email.message_id} from IMAP: {e}")

