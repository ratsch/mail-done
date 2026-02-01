#!/usr/bin/env python3
"""
Production Email Processor - Apply Rules to Inbox

This script processes emails and applies classification rules, VIP detection,
and actions (move, color, label) to your IMAP inbox.

Phase 1: Rule-based classification + VIP detection
Phase 2: Will add AI classification, vector search, etc.

Usage:
    # Dry run (show what would happen, no changes)
    python3.11 process_inbox.py --dry-run --limit 10
    
    # Process new emails only
    python3.11 process_inbox.py --new-only
    
    # Process specific number of emails
    python3.11 process_inbox.py --limit 100
    
    # Process all emails (careful!)
    python3.11 process_inbox.py --all
    
    # Execute actions only on already-processed emails (skip AI, embeddings)
    python3.11 process_inbox.py --actions-only --limit 100
    
    # Process folder and all subfolders recursively
    python3.11 process_inbox.py --folder INBOX --recursive --limit 100
    
Options:
    --dry-run           Show what would happen without making changes
    --new-only          Process only UNSEEN emails (default: all emails)
    --limit N           Process only N emails
    --all               Process ALL emails in folder
    --folder NAME       Which folder to process (default: INBOX)
    --recursive         Process folder and all subfolders recursively (e.g., INBOX processes INBOX, INBOX/Subfolder, etc.)
    --skip-vip          Skip VIP detection
    --skip-rules        Skip rule-based classification
    --skip-ai           Skip AI classification
    --actions-only      Execute actions ONLY on already-processed emails (uses existing ai_category from DB)
    --use-two-stage     Use two-stage classifier (gpt-5-mini → gpt-5.1) instead of one-stage (gpt-5-mini, default)
    --reprocess         Rerun AI classification on all emails (useful after model upgrades)
    --skip-embeddings   Skip vector embedding generation (default: ON)
    --skip-tracking     Skip response tracking (default: ON)
    --generate-drafts   Generate draft replies from AI answer_options (default: OFF)
    --verbose           Show detailed processing info
"""

import asyncio
import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional
from collections import Counter
from dotenv import load_dotenv
from sqlalchemy.orm import Session
import threading

# Load environment variables FIRST (before any imports that need them)
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose HTTP logging from OpenAI/httpx (only show errors)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from backend.core.email.imap_monitor import IMAPMonitor, IMAPConfig
from backend.core.email.processor import EmailProcessor
from backend.core.email.preprocessor import EmailPreprocessor
from backend.core.email.rule_classifier import RuleBasedClassifier
from backend.core.email.vip_manager import VIPManager
from backend.core.email.models import ProcessedEmail, EmailAction
from backend.core.email.email_filter import EmailFilter
from backend.core.paths import get_config_path

# Phase 2: Database and AI
try:
    from backend.core.database import init_db, get_db
    from backend.core.database.repository import EmailRepository
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("Database module not available - running without database storage")

try:
    from backend.core.ai.classifier import AIClassifier
    from backend.core.ai.unified_two_stage_classifier import UnifiedTwoStageClassifier
    from backend.core.ai.config_constants import DEFAULT_MODEL, TWO_STAGE_FAST_MODEL, TWO_STAGE_DETAILED_MODEL
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    DEFAULT_MODEL = "gpt-5-mini"
    TWO_STAGE_FAST_MODEL = "gpt-5-mini"
    TWO_STAGE_DETAILED_MODEL = "gpt-5.1"
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("AI classifier not available - running without AI classification")

# Phase 3: Advanced features
try:
    from backend.core.search.embeddings import EmbeddingGenerator
    from backend.core.tracking.response_tracker import ResponseTracker
    from backend.core.replies.draft_manager import DraftManager
    EMBEDDINGS_AVAILABLE = True
    TRACKING_AVAILABLE = True
    DRAFTS_AVAILABLE = True
except ImportError as e:
    EMBEDDINGS_AVAILABLE = False
    TRACKING_AVAILABLE = False
    DRAFTS_AVAILABLE = False
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Phase 3 features not available: {e}")

# Inquiry Handler (for #application #info emails)
try:
    from backend.core.email.inquiry_processor import InquiryProcessor
    from backend.core.email.inquiry_handler import is_inquiry_email
    from backend.core.email.imap_drafts import IMAPDraftsManager
    INQUIRY_HANDLING_AVAILABLE = True
except ImportError as e:
    INQUIRY_HANDLING_AVAILABLE = False
    logger.warning(f"Inquiry handling not available: {e}")


class EmailProcessingPipeline:
    """
    Production email processing pipeline.
    
    Phase 1: Preprocessing → VIP Detection → Rule Classification → Action Execution
    Phase 2: Will add AI Classification, Vector Search, Reply Generation
    """
    
    def __init__(self,
                 account_id: str = "work",
                 allow_cross_account_moves: bool = False,
                 preprocessing_rules: Optional[str] = None,
                 classification_rules: Optional[str] = None,
                 vip_config: Optional[str] = None,
                 dry_run: bool = True,
                 use_database: bool = True,
                 use_ai: bool = True,
                 use_two_stage: bool = False,
                 reprocess: bool = False,
                 generate_embeddings: bool = True,
                 track_responses: bool = True,
                 generate_drafts: bool = False,
                 parallel_embeddings: int = 1,
                 safe_move: bool = True,
                 create_folders: bool = False,
                 actions_only: bool = False,
                 imap_timeout: int = None):
        """
        Initialize processing pipeline.
        
        Args:
            account_id: Account nickname (e.g., 'work', 'personal', 'work_archive')
            allow_cross_account_moves: If True, allow rules to move emails between accounts
            preprocessing_rules: Path to preprocessing YAML config
            classification_rules: Path to classification YAML config
            vip_config: Path to VIP YAML config
            dry_run: If True, only log actions without executing them
            use_database: If True, store emails and metadata in database (Phase 2)
            use_ai: If True, use AI classification for unmatched emails (Phase 2)
            use_two_stage: If True, use two-stage classifier (gpt-5-mini → gpt-5.1), else one-stage (gpt-5-mini)
            reprocess: If True, rerun AI classification even for previously processed emails
            generate_embeddings: If True, generate vector embeddings for semantic search (Phase 3, default: ON)
            track_responses: If True, track which emails need replies (Phase 3, default: ON)
            generate_drafts: If True, generate draft replies (Phase 3, default: OFF)
            parallel_embeddings: Number of parallel workers for email processing (1-20, default: 1)
                                Note: Each worker processes emails concurrently (AI + embeddings + DB)
            actions_only: If True, skip AI/embeddings/tracking and only execute actions based on existing classifications
            imap_timeout: IMAP connection/read timeout in seconds (default: from IMAP_TIMEOUT env or 120s)
        """
        self.account_id = account_id
        self.allow_cross_account_moves = allow_cross_account_moves
        self.dry_run = dry_run
        self.actions_only = actions_only
        self.use_database = use_database and DATABASE_AVAILABLE
        
        # IMAP timeout: use provided value, else env var, else 120s default (30s was too short for slow servers)
        self.imap_timeout = imap_timeout if imap_timeout is not None else int(os.getenv('IMAP_TIMEOUT', '120'))
        
        # Warn if database is disabled (will reprocess all emails)
        if not self.use_database and not dry_run:
            logger.warning("⚠️  Database disabled: All emails will be reprocessed (no skipping of already-processed emails)")
            logger.warning("   Use --skip-database only for testing. For production, enable database to avoid reprocessing.")
        self.use_ai = use_ai and AI_AVAILABLE
        self.use_two_stage = use_two_stage
        self.reprocess = reprocess
        self.parallel_workers = max(1, min(50, parallel_embeddings))  # Clamp to 1-50
        self.safe_move = safe_move
        self.create_folders = create_folders
        
        # Async lock for IMAP operations (ensure thread-safe access in parallel mode)
        self.imap_lock = asyncio.Lock() if self.parallel_workers > 1 else None
        
        # Phase 3 features
        # In actions-only mode, skip all expensive operations
        if self.actions_only:
            self.generate_embeddings = False
            self.track_responses = False
            self.generate_drafts = False
            self.use_ai = False  # Skip AI processing, use existing classifications
        else:
            self.generate_embeddings = generate_embeddings and EMBEDDINGS_AVAILABLE and self.use_database
            self.track_responses = track_responses and TRACKING_AVAILABLE and self.use_database
            self.generate_drafts = generate_drafts and DRAFTS_AVAILABLE and self.use_database
        
        # Initialize database if enabled
        if self.use_database:
            try:
                init_db()
                logger.info("✅ Database initialized for metadata storage")
            except Exception as e:
                logger.warning(f"Database initialization failed: {e}")
                self.use_database = False
        
        # Initialize AI classifier if enabled
        self.ai_classifier = None
        if self.use_ai:
            try:
                if self.use_two_stage:
                    self.ai_classifier = UnifiedTwoStageClassifier(
                        fast_model=TWO_STAGE_FAST_MODEL,
                        detailed_model=TWO_STAGE_DETAILED_MODEL
                    )
                    logger.info(f"✅ Two-stage AI classifier initialized ({TWO_STAGE_FAST_MODEL} → {TWO_STAGE_DETAILED_MODEL})")
                else:
                    self.ai_classifier = AIClassifier(model=DEFAULT_MODEL)
                    logger.info("✅ AI classifier initialized (gpt-5-mini)")
            except Exception as e:
                logger.warning(f"AI classifier initialization failed: {e}")
                self.use_ai = False
        
        # Initialize Phase 3 components
        self.embedding_generator = None
        self.response_tracker = None
        self.draft_manager = None
        
        if self.generate_embeddings:
            try:
                # Use text-embedding-3-large for best quality (only $8.54/year for 109K emails)
                embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
                self.embedding_generator = EmbeddingGenerator(model=embedding_model)
                if self.parallel_workers > 1:
                    logger.info(f"✅ Embedding generator initialized: {embedding_model} (🚀 parallel workers: {self.parallel_workers})")
                else:
                    logger.info(f"✅ Embedding generator initialized: {embedding_model}")
            except Exception as e:
                logger.warning(f"Embedding generator initialization failed: {e}")
                self.generate_embeddings = False
        
        if self.track_responses:
            logger.info("✅ Response tracking enabled")
        
        if self.generate_drafts:
            logger.info("✅ Draft generation enabled")
        
        # Initialize components
        self.processor = EmailProcessor()
        
        # Initialize AccountManager and CrossAccountMoveService for cross-account moves
        self.account_manager = None
        self.cross_account_service = None
        if self.allow_cross_account_moves:
            try:
                from backend.core.accounts.manager import AccountManager
                from backend.core.email.cross_account_move import CrossAccountMoveService
                self.account_manager = AccountManager()
                self.cross_account_service = CrossAccountMoveService(
                    self.account_manager,
                    dry_run=self.dry_run
                )
                logger.info(f"✅ Cross-account moves enabled for account: {self.account_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize cross-account move service: {e}")
                self.allow_cross_account_moves = False
        
        # Load preprocessing rules
        self.preprocessor = None
        if preprocessing_rules and Path(preprocessing_rules).exists():
            self.preprocessor = EmailPreprocessor.from_yaml(preprocessing_rules)
            print(f"✅ Loaded preprocessing rules from {preprocessing_rules}")
        
        # Load classification rules
        self.classifier = None
        self.junk_folders = []  # Folders where "keep" rescues to INBOX
        if classification_rules and Path(classification_rules).exists():
            self.classifier = RuleBasedClassifier.from_yaml(classification_rules)
            print(f"✅ Loaded {len(self.classifier.rules)} classification rules")
            
            # Load junk_folders config from the same file
            import yaml
            with open(classification_rules, 'r') as f:
                rules_config = yaml.safe_load(f)
                if rules_config and 'config' in rules_config:
                    self.junk_folders = rules_config['config'].get('junk_folders', [])
                    if self.junk_folders:
                        print(f"   Junk folders configured: {', '.join(self.junk_folders)}")
                    
                    # Load categories to skip when processing junk folders
                    # (e.g., spam, marketing - these stay in Junk, not rescued)
                    self.junk_skip_categories = rules_config['config'].get('junk_skip_categories', [])
                    if self.junk_skip_categories:
                        print(f"   Junk skip categories: {', '.join(self.junk_skip_categories)}")
        
        # Default skip categories if not configured
        if not hasattr(self, 'junk_skip_categories'):
            self.junk_skip_categories = ['spam', 'marketing', 'social-media']
        
        # Track current folder being processed (for context-aware "keep" behavior)
        self._current_folder = 'INBOX'
        
        # Load VIP manager
        self.vip_manager = None
        if vip_config and Path(vip_config).exists():
            self.vip_manager = VIPManager(vip_config)
            stats = self.vip_manager.get_vip_stats()
            total_vips = sum(stats.values())
            print(f"✅ Loaded {total_vips} VIP senders (Urgent: {stats['urgent']}, High: {stats['high']}, Medium: {stats['medium']})")
        
        # Load AI category actions mapping
        self.ai_category_actions = {}
        self.ai_allowed_folders = set()  # Whitelist of approved folders
        ai_actions_path = get_config_path("ai_category_actions.yaml")
        if ai_actions_path and ai_actions_path.exists():
            import yaml
            with open(ai_actions_path, 'r') as f:
                ai_actions_config = yaml.safe_load(f)
                if ai_actions_config:
                    self.ai_category_actions = ai_actions_config
                    # Extract whitelist of allowed folders from mapping
                    for category, config in ai_actions_config.items():
                        if isinstance(config, dict) and config.get('action') == 'move':
                            folder = config.get('folder')
                            if folder:
                                self.ai_allowed_folders.add(folder)
                    
                    num_categories = len([k for k in ai_actions_config.keys() if k != 'default'])
                    num_folders = len(self.ai_allowed_folders)
                    print(f"✅ Loaded AI category actions for {num_categories} categories")
                    print(f"   Whitelisted {num_folders} approved folders for AI moves")
        
        # Statistics
        self.stats = {
            'processed': 0,
            'vip_detected': 0,
            'rule_matched': 0,
            'ai_classified': 0,
            'ai_failures': 0,
            'ai_reprocessed': 0,
            'db_stored': 0,
            'db_failures': 0,
            'embeddings_generated': 0,
            'embeddings_skipped': 0,
            'embeddings_failures': 0,
            'responses_tracked': 0,
            'drafts_generated': 0,
            'drafts_failures': 0,
            'actions_taken': Counter(),
            'folders_used': Counter(),
            'colors_applied': Counter(),
            'errors': []
        }
        
        if dry_run:
            print(f"\n🔍 DRY RUN MODE - No changes will be made to IMAP server")
        else:
            print(f"\n⚠️  LIVE MODE - Changes will be applied to IMAP server")
        
        if reprocess:
            print(f"🔄 REPROCESS MODE - AI will reclassify all emails, even if previously processed")
        
        if self.parallel_workers > 1:
            print(f"🚀 PARALLEL MODE - Processing {self.parallel_workers} emails concurrently per batch")
        else:
            print(f"📝 SEQUENTIAL MODE - Processing emails one at a time")
    
    async def process_email(self,
                           email: ProcessedEmail,
                           imap: IMAPMonitor) -> Dict:
        """
        Process a single email through the pipeline.
        
        Pipeline:
        1. Preprocessing (extract original sender from forwards)
        2. VIP Detection (check if sender is VIP)
        3. Rule-based Classification (apply rules)
        4. [Phase 2: AI Classification]
        5. Action Execution (move, color, label)
        
        Args:
            email: Processed email to classify
            imap: IMAP monitor for executing actions
            
        Returns:
            Processing result dictionary
        """
        result = {
            'uid': email.uid,
            'subject': email.subject,
            'from': email.from_address,
            'from_name': email.from_name,
            'date': email.date,
            'preprocessed': False,
            'preprocessing_changed': False,  # Only True if preprocessing actually changed something
            'vip_level': None,
            'rule_matched': None,
            'actions': [],
            'executed': [],
            'dry_run_actions': [],  # Actions that would be executed in dry-run mode
            'ai_classification': None,
            'ai_time': None,
            'cache_hit': False,
            'cache_stats': None,
            'error': None
        }
        
        # Initialize two_stage_metadata for closure scope (used later in sync_db_operations)
        two_stage_metadata = None
        
        # Step 1: Preprocessing (extract original sender from forwarded emails)
        if self.preprocessor:
            original_from = email.from_address
            original_from_name = email.from_name
            preprocessed = self.preprocessor.preprocess(email, email.raw_headers)
            if preprocessed.was_preprocessed:
                result['preprocessed'] = True
                result['preprocessing_changed'] = True  # Only log when it changes something
                result['original_from'] = original_from
                result['from'] = preprocessed.from_address
                result['from_name'] = preprocessed.from_name  # Update from_name if changed
                email = preprocessed  # Use preprocessed version
        
        # Step 1.5: Inquiry Detection (Stage 0 - before VIP/rules/AI)
        # Check for #info + #application tags in subject line
        if INQUIRY_HANDLING_AVAILABLE and is_inquiry_email(email.subject):
            logger.info(f"📧 Inquiry detected for email {email.uid}: {email.subject[:50]}...")
            result['is_inquiry'] = True
            
            # Process through inquiry pipeline
            try:
                # Get database session and repository if available
                inquiry_db_session = None
                inquiry_repo = None
                inquiry_db_email = None
                
                if self.use_database:
                    inquiry_db_session = next(get_db())
                    inquiry_repo = EmailRepository(inquiry_db_session)
                    from backend.core.database.models import Email as EmailModel
                    inquiry_db_email = inquiry_db_session.query(EmailModel).filter(
                        EmailModel.message_id == email.message_id,
                        EmailModel.account_id == self.account_id
                    ).first()
                
                # Create IMAP drafts manager using current account's config
                imap_drafts_manager = None
                if not self.dry_run and hasattr(self, '_current_imap_config'):
                    cfg = self._current_imap_config
                    imap_drafts_manager = IMAPDraftsManager(
                        imap_host=cfg.host,
                        imap_username=cfg.username,
                        imap_password=cfg.password
                    )
                
                # Process inquiry
                inquiry_processor = InquiryProcessor(
                    ai_classifier=self.ai_classifier if self.use_ai else None,
                    db_session=inquiry_db_session,
                    dry_run=self.dry_run
                )
                
                inquiry_result = await inquiry_processor.process_inquiry(
                    email=email,
                    imap_drafts_manager=imap_drafts_manager,
                    repo=inquiry_repo,
                    db_email=inquiry_db_email
                )
                
                # Update result
                result.update(inquiry_result)
                
                if inquiry_result.get('skip_normal_pipeline', False):
                    # Inquiry was valid and processed - skip normal pipeline
                    result['category'] = inquiry_result.get('ai_category', 'inquiry-unknown')
                    result['actions'].append(f"Inquiry: Draft created → Drafts folder")
                    if inquiry_result.get('moved_to_folder'):
                        result['actions'].append(f"Inquiry: Move → {inquiry_result['moved_to_folder']}")
                    self.stats['inquiry_processed'] = self.stats.get('inquiry_processed', 0) + 1
                    
                    if inquiry_db_session:
                        inquiry_db_session.close()
                    
                    return result  # Skip rest of pipeline
                else:
                    # Not a valid inquiry - continue with normal pipeline
                    logger.info(f"Email {email.uid} not valid inquiry, routing to normal pipeline")
                    if inquiry_db_session:
                        inquiry_db_session.close()
                    
            except Exception as e:
                logger.error(f"Inquiry processing failed for {email.uid}: {e}", exc_info=True)
                result['inquiry_error'] = str(e)
                # Continue with normal pipeline as fallback
        
        # Step 2: VIP Detection
        vip_action = None
        if self.vip_manager:
            vip_info = self.vip_manager.check_vip(email)
            if vip_info:
                result['vip_level'] = vip_info.level
                self.stats['vip_detected'] += 1
                
                # Create VIP action (flag for visibility, color tracked for Phase 2 database)
                vip_action = EmailAction(
                    type='color',  # Keep color type for Phase 2 metadata
                    color=vip_info.color,
                    labels=['VIP', vip_info.level.upper()]
                )
                result['actions'].append(f"VIP: Flag (level: {vip_info.level}, color for DB: {vip_info.color})")
        
        # Step 3: Rule-based Classification
        rule_action = None
        rule_application_source = None
        if self.classifier:
            classification = self.classifier.classify(email)
            if classification:
                category, action, application_source = classification
                result['rule_matched'] = action
                result['category'] = category.value
                result['application_source'] = application_source  # For filtering in portal
                self.stats['rule_matched'] += 1
                rule_action = action
                rule_application_source = application_source
                
                # Describe action
                if action.type == 'move':
                    result['actions'].append(f"Rule: Move → {action.folder}")
                elif action.type == 'color':
                    result['actions'].append(f"Rule: Color {action.color}")
                elif action.type == 'label':
                    result['actions'].append(f"Rule: Label {action.labels}")
                elif action.type == 'archive':
                    result['actions'].append(f"Rule: Archive")
        
        # Step 4: AI Classification (Phase 2)
        # Check if email already has AI classification in database
        already_classified = False
        existing_ai_category = None
        if self.use_database and (not self.reprocess or self.actions_only):
            try:
                from backend.core.database.models import Email as EmailModel, EmailMetadata
                session = next(get_db())
                logger.debug(f"Checking if email {email.uid} (message_id: {email.message_id}) already classified...")
                existing_email = session.query(EmailModel).filter(
                    EmailModel.message_id == email.message_id,
                    EmailModel.account_id == self.account_id
                ).first()
                if existing_email:
                    logger.debug(f"Found email in database with id: {existing_email.id}")
                    # Check if email has AI classification in metadata
                    metadata = session.query(EmailMetadata).filter(
                        EmailMetadata.email_id == existing_email.id
                    ).first()
                    if metadata and metadata.ai_category:
                        already_classified = True
                        existing_ai_category = metadata.ai_category
                        logger.debug(f"Email {email.uid} already has AI category: {existing_ai_category}")
                    else:
                        logger.debug(f"Email {email.uid} has NO AI classification yet")
                else:
                    logger.debug(f"Email {email.uid} NOT found in database")
                session.close()
            except Exception as e:
                logger.warning(f"Could not check existing classification: {e}")
        
        # Run AI if:
        # - AI is enabled AND
        # - Not in actions-only mode AND
        # - Email not yet AI-classified (or reprocess flag is set) AND
        # - (No rule matched OR reprocess flag is set - to allow enriching rule matches with AI)
        ai_classification = None
        should_run_ai = (
            not self.actions_only and
            self.use_ai and 
            self.ai_classifier and 
            (self.reprocess or not already_classified)
        )
        is_reprocessing = self.reprocess and (rule_action or already_classified)
        
        # In actions-only mode, create a mock AI classification from existing category
        if self.actions_only and existing_ai_category:
            from backend.core.ai.classifier import AIClassificationResult
            ai_classification = AIClassificationResult(
                category=existing_ai_category,
                confidence=1.0,  # Use existing classification
                reasoning="Using existing classification from database (--actions-only mode)",
                needs_reply=False,
                urgency="normal",
                urgency_score=5,
                urgency_reason="Using existing classification (--actions-only mode)",
                summary=f"Email classified as {existing_ai_category} (using existing classification)",
                sentiment="neutral"
            )
            logger.debug(f"Actions-only mode: Using existing AI category '{existing_ai_category}'")
        
        if should_run_ai:
            try:
                import time
                ai_start_time = time.time()
                
                # Enable cost tracking for AI classification
                if self.use_database:
                    session = next(get_db())
                    if hasattr(self.ai_classifier, 'set_cost_tracker_session'):
                        self.ai_classifier.set_cost_tracker_session(session, source="cli")
                
                # Classify with unified interface
                # Both AIClassifier and UnifiedTwoStageClassifier return AIClassificationResult
                ai_classification = await self.ai_classifier.classify(email)
                
                # Track AI processing time
                ai_time = time.time() - ai_start_time
                result['ai_time'] = ai_time
                
                # Note: Cache stats are logged at debug level by the classifier
                # Cache info will be shown in card output if available from usage stats
                
                # Extract curated sender name (use AI result if available, else fallback to separate LLM call)
                curated_sender_name = None
                if ai_classification and ai_classification.curated_sender_name:
                    curated_sender_name = ai_classification.curated_sender_name
                    logger.debug(f"Using curated sender name from classification: {curated_sender_name}")
                    result['curated_sender_name'] = curated_sender_name
                else:
                    # Fallback: Extract curated sender name using separate LLM call
                    try:
                        from backend.core.email.name_extraction import extract_curated_sender_name
                        curated_sender_name = await extract_curated_sender_name(email)
                        if curated_sender_name:
                            logger.debug(f"Extracted curated sender name (fallback): {curated_sender_name}")
                            result['curated_sender_name'] = curated_sender_name
                    except Exception as e:
                        logger.debug(f"Failed to extract curated sender name: {e}")
                
                # Critical Step: For applications, ensure applicant_name is populated
                # If applicant_name is missing but we have a curated_sender_name, assume sender IS the applicant
                # (Unless it's a forwarded/recommendation email, which should be work-colleague, not application-*)
                if ai_classification and ai_classification.category.startswith('application-'):
                    if not ai_classification.applicant_name and curated_sender_name:
                        ai_classification.applicant_name = curated_sender_name
                        logger.info(f"Populated empty applicant_name with curated_sender_name: {curated_sender_name}")
                
                # Get two-stage metadata if available (for DB storage)
                stage_2_triggered = False
                if self.use_two_stage and hasattr(self.ai_classifier, 'get_last_detailed_result'):
                    # Get cached metadata from last classify() call (no re-classification!)
                    detailed_result = self.ai_classifier.get_last_detailed_result()
                    if detailed_result:
                        two_stage_metadata = detailed_result.get('two_stage_metadata')
                        stage_2_triggered = detailed_result.get('stage_2_triggered', False)
                
                # Commit cost tracking immediately
                if self.use_database:
                    session.commit()
                    session.close()
                if ai_classification:
                    result['ai_classification'] = {
                        'category': ai_classification.category,
                        'confidence': ai_classification.confidence,
                        'summary': ai_classification.summary,
                        'urgency': ai_classification.urgency,
                        'urgency_score': getattr(ai_classification, 'urgency_score', None),
                        'sentiment': ai_classification.sentiment,
                        'needs_reply': ai_classification.needs_reply,
                        'is_cold_email': ai_classification.is_cold_email,
                        'deadline': getattr(ai_classification, 'deadline', None),
                        'event_date': getattr(ai_classification, 'event_date', None),
                        'location': getattr(ai_classification, 'location', None),
                        'time_commitment_hours': getattr(ai_classification, 'time_commitment_hours', None),
                        'relevance_score': getattr(ai_classification, 'relevance_score', None),
                        'prestige_score': getattr(ai_classification, 'prestige_score', None),
                        'applicant_name': getattr(ai_classification, 'applicant_name', None),
                        'applicant_institution': getattr(ai_classification, 'applicant_institution', None),
                        'scientific_excellence_score': getattr(ai_classification, 'scientific_excellence_score', None),
                        'recommendation_score': getattr(ai_classification, 'recommendation_score', None)
                    }
                    if stage_2_triggered:
                        # Include reason for Stage 2 trigger
                        reason = detailed_result.get('stage_2_reason', 'unknown')
                        stage_2_error = detailed_result.get('stage_2_error')
                        if stage_2_error:
                            result['actions'].append(f"AI: {ai_classification.category} (confidence: {ai_classification.confidence:.2f}) [Stage 2: {reason}, error: {stage_2_error}]")
                        else:
                            result['actions'].append(f"AI: {ai_classification.category} (confidence: {ai_classification.confidence:.2f}) [Stage 2: {reason}]")
                    else:
                        result['actions'].append(f"AI: {ai_classification.category} (confidence: {ai_classification.confidence:.2f})")
                    self.stats['ai_classified'] += 1
                    if is_reprocessing:
                        self.stats['ai_reprocessed'] += 1
                        result['reprocessed'] = True
            except Exception as e:
                logger.warning(f"AI classification failed for {email.uid}: {e}")
                result['error'] = f"AI classification failed: {str(e)}"
                result['ai_error'] = str(e)
                self.stats['ai_failures'] += 1
        
        # Step 5-8: Database Operations (run in a separate thread to avoid blocking asyncio event loop)
        db_email = None
        if self.use_database:
            
            def sync_db_operations():
                """Encapsulates all synchronous database operations for a single email."""
                _db_email = None
                session = next(get_db())
                try:
                    # Enable cost tracking for this session
                    if self.ai_classifier:
                        self.ai_classifier.set_cost_tracker_session(session, source="cli")
                    if self.embedding_generator:
                        self.embedding_generator.set_cost_tracker_session(session, source="cli")
                    
                    repo = EmailRepository(session)
                    
                    # Step 5: Store Email Record
                    _db_email, is_new = repo.get_or_create_email(email, account_id=self.account_id)
                    if not _db_email:
                        raise RuntimeError("Failed to get or create email record in database.")

                    result['db_new'] = is_new
                    
                    # Store metadata (VIP level, color intent, AI classification, two-stage info)
                    # In actions-only mode, don't update metadata (use existing)
                    vip_level = result.get('vip_level')
                    intended_color = vip_action.color if vip_action and hasattr(vip_action, 'color') else None
                    if not self.actions_only:
                        # Get curated sender name from result (extracted during classification)
                        curated_sender_name = result.get('curated_sender_name')
                        # Get rule category if a rule matched (takes precedence over AI category)
                        rule_category = result.get('category') if rule_action else None
                        # Get application source from rule (for filtering in portal)
                        application_source = result.get('application_source') if rule_action else None
                        repo.store_metadata(_db_email, vip_level, intended_color, ai_result=ai_classification, 
                                           two_stage_metadata=two_stage_metadata,
                                           curated_sender_name=curated_sender_name,
                                           rule_category=rule_category,
                                           application_source=application_source)
                    
                    # Store rule-based classification if matched
                    if rule_action:
                        repo.store_classification(
                            _db_email, classifier_type="rule", category=result.get('category', 'unknown'),
                            confidence=1.0, reasoning=f"Matched rule for {rule_action.type}"
                        )
                    
                    # Store AI classification if available (skip in actions-only mode - using existing)
                    if ai_classification and not self.actions_only:
                        repo.store_classification(
                            _db_email, classifier_type="ai", category=ai_classification.category,
                            confidence=ai_classification.confidence, reasoning=ai_classification.reasoning
                        )
                    
                    # Update sender history (skip in parallel mode to avoid lock contention)
                    if self.parallel_workers == 1:
                        repo.update_sender_history(_db_email)
                    
                    # Step 6: Generate Embeddings (Phase 3)
                    if self.generate_embeddings and self.embedding_generator:
                        self._sync_generate_embedding(session, _db_email, result)
                    
                    # Step 7: Track Responses (Phase 3)
                    if self.track_responses and ai_classification:
                        self._sync_track_response(session, _db_email, ai_classification, result)
                        
                    # Step 8: Generate Drafts (Phase 3)
                    if self.generate_drafts and ai_classification:
                        self._sync_generate_drafts(session, _db_email, email, ai_classification, result)

                    # All operations successful, commit the transaction
                    session.commit()
                    return _db_email
                
                except Exception as db_e:
                    logger.error(f"Database operations failed for {email.uid}: {db_e}")
                    session.rollback()
                    result['db_error'] = str(db_e)
                    self.stats['db_failures'] += 1
                    return None
                finally:
                    session.close()

            db_email = await asyncio.to_thread(sync_db_operations)

            if db_email:
                self.stats['db_stored'] += 1
                result['db_stored'] = True
        
        # Step 9: Execute Actions
        if vip_action:
            executed = await self._execute_action(email.uid, vip_action, imap)
            if executed:
                result['executed'].append(f"Applied VIP color {vip_action.color}")
            elif self.dry_run:
                result['dry_run_actions'].append(f"Would apply VIP color {vip_action.color}")
        
        if rule_action:
            executed, cross_account_info = await self._execute_action_with_info(email.uid, rule_action, imap)
            if executed:
                if cross_account_info:
                    result['executed'].append(f"✅ Cross-account move: {cross_account_info}")
                    result['cross_account_move'] = cross_account_info
                else:
                    result['executed'].append(f"Applied rule action: {rule_action.type}")
            elif not executed and not self.dry_run:
                # Action failed (e.g., move timeout) - track for sync state
                result['move_failed'] = True
                result['action_error'] = f"Failed to execute rule action: {rule_action.type}"
                logger.warning(f"Rule action failed for UID {email.uid}: {rule_action.type}")
            elif self.dry_run:
                # In dry-run, show what would happen even if not executed
                # Use cross_account_info if available (from _execute_action_with_info)
                if cross_account_info:
                    if cross_account_info.startswith('RESCUE:'):
                        # Junk folder rescue - "keep" action would move to INBOX
                        rescue_info = cross_account_info.replace('RESCUE:', '')
                        result['dry_run_actions'].append(f"🛟 Would rescue: {rescue_info}")
                    elif cross_account_info.startswith('ALREADY_IN:'):
                        # Already in destination folder - no action needed
                        folder = cross_account_info.replace('ALREADY_IN:', '')
                        result['dry_run_actions'].append(f"✓ Already in {folder} (no action needed)")
                    elif cross_account_info.startswith('SKIPPED:'):
                        # Cross-account move would be skipped (flag not set)
                        skipped_info = cross_account_info.replace('SKIPPED:', '')
                        result['dry_run_actions'].append(f"Would move to {skipped_info} (cross-account move SKIPPED - --allow-cross-account-moves not set)")
                    else:
                        result['dry_run_actions'].append(f"Would move to {cross_account_info} (cross-account)")
                        result['cross_account_move'] = cross_account_info
                elif rule_action.type == 'move' and rule_action.folder:
                    result['dry_run_actions'].append(f"Would move to {rule_action.folder}")
                elif rule_action.type == 'color':
                    result['dry_run_actions'].append(f"Would apply color {rule_action.color}")
                elif rule_action.type == 'label':
                    result['dry_run_actions'].append(f"Would add labels: {', '.join(rule_action.labels)}")
                elif rule_action.type == 'archive':
                    result['dry_run_actions'].append(f"Would archive")
                elif rule_action.type == 'keep':
                    # Context-aware message for "keep" action
                    if self._is_junk_folder():
                        result['dry_run_actions'].append(f"🛟 Would rescue to INBOX (from {self._current_folder})")
                    else:
                        # Not in junk: "keep" means no action (leave where it is)
                        if self._current_folder and self._current_folder != 'INBOX':
                            result['dry_run_actions'].append(f"Would keep in {self._current_folder} (no action)")
                        else:
                            result['dry_run_actions'].append(f"Would keep in INBOX (no action)")
        
        # Execute AI-based action (if AI classified and no rule action taken)
        # NOTE: In junk folders, we skip AI-based moves - only explicit "keep" rules rescue emails.
        #       This prevents spam/marketing from being moved to MD/Marketing etc.
        #       Emails stay in Junk unless a rule explicitly rescues them.
        if ai_classification and not rule_action and self.ai_category_actions:
            ai_action = self._create_action_from_ai_category(ai_classification.category)
            if ai_action:
                # In junk folders: skip AI moves for spam/marketing, allow others (receipts, notifications)
                # This rescues legitimate emails while keeping actual spam in Junk
                should_skip = False
                if self._is_junk_folder() and ai_action.type == 'move':
                    # Check if this category should be skipped (spam, marketing, etc.)
                    category_base = ai_classification.category.split('-')[0]  # e.g., "receipt-online" → "receipt"
                    if (ai_classification.category in self.junk_skip_categories or 
                        category_base in self.junk_skip_categories):
                        should_skip = True
                
                if should_skip:
                    if self.dry_run:
                        result['dry_run_actions'].append(
                            f"⏭️ Skipped: Would move to {ai_action.folder} (AI: {ai_classification.category}) "
                            f"- category in junk_skip_categories, stays in Junk"
                        )
                    else:
                        logger.debug(f"Skipping AI move in junk folder (blocked category): {ai_classification.category} → {ai_action.folder}")
                else:
                    # Normal execution (not in junk folder, or non-move action)
                    executed, cross_account_info = await self._execute_action_with_info(email.uid, ai_action, imap)
                    if executed:
                        if cross_account_info:
                            result['executed'].append(f"✅ Cross-account move: {cross_account_info}")
                            result['cross_account_move'] = cross_account_info
                        else:
                            result['executed'].append(f"Applied AI action: {ai_action.type} ({ai_classification.category})")
                    elif not executed and not self.dry_run:
                        # AI action failed (e.g., move timeout) - track for sync state
                        result['move_failed'] = True
                        result['action_error'] = f"Failed to execute AI action: {ai_action.type} ({ai_classification.category})"
                        logger.warning(f"AI action failed for UID {email.uid}: {ai_action.type}")
                    elif self.dry_run:
                        # In dry-run, show what would happen even if not executed
                        # Use cross_account_info if available (from _execute_action_with_info)
                        if cross_account_info:
                            if cross_account_info.startswith('ALREADY_IN:'):
                                # Already in destination folder - no action needed
                                folder = cross_account_info.replace('ALREADY_IN:', '')
                                result['dry_run_actions'].append(f"✓ Already in {folder} (AI: {ai_classification.category}, no action needed)")
                            elif cross_account_info.startswith('SKIPPED:'):
                                # Cross-account move would be skipped (flag not set)
                                skipped_info = cross_account_info.replace('SKIPPED:', '')
                                result['dry_run_actions'].append(f"Would move to {skipped_info} (cross-account move SKIPPED - --allow-cross-account-moves not set, AI: {ai_classification.category})")
                            else:
                                result['dry_run_actions'].append(f"Would move to {cross_account_info} (cross-account, AI: {ai_classification.category})")
                                result['cross_account_move'] = cross_account_info
                        elif ai_action.type == 'move' and ai_action.folder:
                            result['dry_run_actions'].append(f"Would move to {ai_action.folder} (AI: {ai_classification.category})")
                        elif ai_action.type == 'color':
                            result['dry_run_actions'].append(f"Would apply color {ai_action.color} (AI: {ai_classification.category})")
                        elif ai_action.type == 'label':
                            result['dry_run_actions'].append(f"Would add labels: {', '.join(ai_action.labels)} (AI: {ai_classification.category})")
                        elif ai_action.type == 'archive':
                            result['dry_run_actions'].append(f"Would archive (AI: {ai_classification.category})")
        
        return result
    
    def _sync_generate_embedding(self, db_session: Session, db_email: "Email", result: Dict):
        """Synchronous helper for embedding generation with smart folder-based skipping."""
        from backend.core.database.models import EmailEmbedding
        from backend.core.database.repository import EmailRepository
        from sqlalchemy.exc import OperationalError, DBAPIError
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Check if we should generate embedding for this folder
                repo = EmailRepository(db_session)
                if not repo.should_generate_embedding(db_email, db_email.folder):
                    self.stats['embeddings_skipped'] += 1
                    result['embedding_skipped'] = 'folder_policy'
                    return
                
                existing_embedding = db_session.query(EmailEmbedding).filter(EmailEmbedding.email_id == db_email.id).first()
                if not existing_embedding or self.force_reembed:
                    embedding, content_hash = self.embedding_generator.generate_embedding(db_email, db_email.email_metadata)
                    if all(x == 0.0 for x in embedding):
                        raise RuntimeError("Embedding generation resulted in a zero vector.")
                    
                    if existing_embedding:
                        existing_embedding.embedding = embedding
                        existing_embedding.content_hash = content_hash
                        existing_embedding.updated_at = datetime.now(timezone.utc)
                    else:
                        new_embedding = EmailEmbedding(
                            email_id=db_email.id, embedding=embedding,
                            embedding_model=self.embedding_generator.model, content_hash=content_hash
                        )
                        db_session.add(new_embedding)
                    
                    self.stats['embeddings_generated'] += 1
                    result['embedding_generated'] = True
                else:
                    self.stats['embeddings_skipped'] += 1
                    result['embedding_skipped'] = 'already_exists'
                return  # Success
                
            except (OperationalError, DBAPIError) as e:
                error_msg = str(e).lower()
                is_connection_error = any(pattern in error_msg for pattern in [
                    'closed the connection', 'connection refused', 'timeout',
                    'server terminated', 'invalid transaction', 'cannot reconnect'
                ])
                
                if is_connection_error and attempt < max_retries - 1:
                    logger.warning(f"Database connection error during embedding (attempt {attempt+1}): {e}")
                    try:
                        db_session.rollback()
                    except Exception:
                        pass
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    continue
                else:
                    logger.warning(f"Embedding generation failed for {result['uid']}: {e}")
                    result['embedding_error'] = str(e)
                    self.stats['embeddings_failures'] += 1
                    try:
                        db_session.rollback()
                    except Exception:
                        pass
                    return
                    
            except Exception as e:
                logger.warning(f"Embedding generation failed for {result['uid']}: {e}")
                result['embedding_error'] = str(e)
                self.stats['embeddings_failures'] += 1
                return

    def _sync_track_response(self, db_session: Session, db_email: "Email", ai_classification: "AIClassificationResult", result: Dict):
        """Synchronous helper for response tracking."""
        try:
            # In parallel mode, always create a new tracker to avoid session conflicts
            # In sequential mode, reuse the tracker for efficiency
            if self.parallel_workers > 1:
                response_tracker = ResponseTracker(db_session)
            else:
                if not self.response_tracker:
                    self.response_tracker = ResponseTracker(db_session)
                else:
                    self.response_tracker.db = db_session
                response_tracker = self.response_tracker
            
            ai_metadata = {
                'ai_category': ai_classification.category, 'needs_reply': ai_classification.needs_reply,
                'urgency': ai_classification.urgency, 'urgency_score': ai_classification.urgency_score,
                'deadline': ai_classification.deadline, 'reply_deadline': ai_classification.reply_deadline,
            }
            tracking = asyncio.run(response_tracker.analyze_and_track(db_email, ai_metadata, result.get('vip_level')))
            
            if tracking:
                self.stats['responses_tracked'] += 1
                result['response_tracked'] = True
                result['needs_reply'] = True
        except Exception as e:
            logger.warning(f"Response tracking failed for {result['uid']}: {e}")
            result['tracking_error'] = str(e)
    
    def _sync_generate_drafts(self, db_session: Session, db_email: "Email", email: ProcessedEmail, ai_classification: "AIClassificationResult", result: Dict):
        """Synchronous helper for draft generation."""
        from backend.core.database.models import ReplyDraft as DBReplyDraft
        try:
            if ai_classification.answer_options:
                for idx, option in enumerate(ai_classification.answer_options, 1):
                    draft = DBReplyDraft(
                        email_id=db_email.id, subject=f"Re: {email.subject}", body=option.get('text', ''),
                        tone=option.get('tone', 'neutral'), option_number=idx, generated_by='ai_classifier',
                        model_used=self.ai_classifier.model if self.ai_classifier else 'unknown',
                        confidence=ai_classification.confidence, reasoning=f"AI-generated {option.get('tone', 'neutral')} response",
                        decision=self._map_tone_to_decision(option.get('tone', 'neutral')),
                        category=ai_classification.category, status='pending'
                    )
                    db_session.add(draft)
                
                self.stats['drafts_generated'] += len(ai_classification.answer_options)
                result['drafts_generated'] = len(ai_classification.answer_options)
        except Exception as e:
            logger.warning(f"Draft generation failed for {result['uid']}: {e}")
            result['draft_error'] = str(e)
            self.stats['drafts_failures'] += 1

    def _map_tone_to_decision(self, tone: str) -> str:
        """Map tone from answer_options to decision type."""
        tone_map = {
            'positive': 'accept',
            'decline': 'decline',
            'inquiry': 'maybe',
            'neutral': 'acknowledge'
        }
        return tone_map.get(tone.lower(), 'acknowledge')
    
    def _is_junk_folder(self, folder: Optional[str] = None) -> bool:
        """
        Check if the given folder (or current folder) is a junk/spam folder.
        
        Used for context-aware "keep" behavior:
        - In junk folders: "keep" rescues emails to INBOX
        - In other folders: "keep" leaves emails in place
        
        Matching is case-insensitive and checks if the folder name or path
        contains any of the configured junk folder names.
        
        Args:
            folder: Folder to check (defaults to current processing folder)
            
        Returns:
            True if folder is a junk/spam folder
        """
        check_folder = folder or self._current_folder
        if not check_folder or not self.junk_folders:
            return False
        
        check_lower = check_folder.lower()
        for junk_folder in self.junk_folders:
            junk_lower = junk_folder.lower()
            # Match exact folder name or as part of path
            if check_lower == junk_lower or check_lower.endswith('/' + junk_lower):
                return True
        return False
    
    def _create_action_from_ai_category(self, category: str) -> Optional[EmailAction]:
        """
        Create an EmailAction from an AI-classified category.
        
        SECURITY: Strict whitelist enforcement
        - Only uses categories explicitly defined in config/ai_category_actions.yaml
        - Only moves to folders approved in the mapping file
        - Rejects any category not in whitelist
        
        Args:
            category: AI classification category (e.g., "invitation-speaking")
            
        Returns:
            EmailAction if category is whitelisted, None otherwise
        """
        if not self.ai_category_actions:
            return None
        
        # SECURITY CHECK 1: Category must be explicitly in mapping (no fuzzy matching)
        if category not in self.ai_category_actions:
            # Try default fallback
            if 'default' in self.ai_category_actions:
                logger.debug(f"AI category '{category}' not in whitelist - using default action")
                action_config = self.ai_category_actions['default']
            else:
                logger.warning(f"AI category '{category}' not in whitelist and no default - REJECTING")
                return None
        else:
            action_config = self.ai_category_actions[category]
        
        if not action_config or not isinstance(action_config, dict):
            logger.error(f"Invalid config for category '{category}' - REJECTING")
            return None
        
        action_type = action_config.get('action')
        if not action_type:
            logger.warning(f"No action type for category '{category}' - REJECTING")
            return None
        
        # Create EmailAction based on type
        if action_type == 'move':
            folder = action_config.get('folder')
            if not folder:
                logger.warning(f"AI category '{category}' has 'move' action but no folder specified - REJECTING")
                return None
            
            # SECURITY CHECK 2: Folder must be in whitelist
            if folder not in self.ai_allowed_folders:
                logger.error(f"SECURITY: Folder '{folder}' not in whitelist for category '{category}' - REJECTING")
                return None
            
            # SECURITY CHECK 3: Validate folder path (no path traversal)
            if '..' in folder or folder.startswith('/') or '\\' in folder:
                logger.error(f"SECURITY: Invalid folder path '{folder}' for category '{category}' - REJECTING")
                return None
            
            # All checks passed - create action
            # Check for cross-account routing (e.g., MD/Personal → personal account)
            target_account = action_config.get('target_account')
            action = EmailAction(type='move', folder=folder, target_account=target_account)
            action._ai_generated = True  # Mark as AI-generated (no auto-create folders)
            return action
        
        elif action_type == 'color':
            color = action_config.get('color')
            if color:
                return EmailAction(type='color', color=color)
            return None
        
        elif action_type == 'label':
            labels = action_config.get('labels', [])
            return EmailAction(type='label', labels=labels)
        
        elif action_type == 'archive':
            return EmailAction(type='archive')
        
        elif action_type == 'keep':
            return None  # No action needed
        
        else:
            logger.warning(f"Unknown action type '{action_type}' for category '{category}'")
            return None
    
    async def _execute_action(self,
                             uid: str,
                             action: EmailAction,
                             imap: IMAPMonitor) -> bool:
        """
        Execute an email action on IMAP server.
        
        Args:
            uid: Email UID
            action: Action to execute
            imap: IMAP monitor
            
        Returns:
            True if action executed successfully
        """
        executed, _ = await self._execute_action_with_info(uid, action, imap)
        return executed
    
    async def _execute_action_with_info(self,
                                       uid: str,
                                       action: EmailAction,
                                       imap: IMAPMonitor) -> tuple[bool, Optional[str]]:
        """
        Execute an email action on IMAP server and return execution status with cross-account info.
        
        Args:
            uid: Email UID
            action: Action to execute
            imap: IMAP monitor
            
        Returns:
            (success: bool, cross_account_info: Optional[str])
            cross_account_info is a formatted string if cross-account move occurred, None otherwise
            In dry-run mode, success=False but cross_account_info may still be populated for display
        """
        if self.dry_run:
            # Dry run: determine what would happen but don't execute
            # Handle "keep" action in junk folders - would rescue to INBOX
            if action.type == 'keep' and self._is_junk_folder():
                return False, f"RESCUE:{self._current_folder} → INBOX"
            
            # Check if move destination equals current folder (no-op)
            if action.type == 'move' and action.folder:
                if action.folder == self._current_folder:
                    return False, f"ALREADY_IN:{action.folder}"
            
            # Return cross-account info if applicable for display purposes
            if action.type == 'move' and action.folder and action.target_account:
                if action.target_account != self.account_id:
                    if self.allow_cross_account_moves:
                        return False, f"{self.account_id} → {action.target_account}: {action.folder}"
                    else:
                        # Return special marker to indicate skipped cross-account move
                        return False, f"SKIPPED:{self.account_id} → {action.target_account}: {action.folder}"
            return False, None
        
        # Acquire IMAP lock for thread-safe operations in parallel mode
        if self.imap_lock:
            async with self.imap_lock:
                return await self._execute_action_locked(uid, action, imap)
        else:
            return await self._execute_action_locked(uid, action, imap)
    
    async def _execute_action_locked(
        self,
        uid: str,
        action: EmailAction,
        imap: IMAPMonitor
    ) -> tuple[bool, Optional[str]]:
        """
        Execute action with IMAP lock already acquired.
        
        Returns:
            (success: bool, cross_account_info: Optional[str])
            cross_account_info is formatted string like "work → personal: MD/Personal" if cross-account move occurred
        
        Note on "keep" action:
            The "keep" action behavior is context-aware based on source folder:
            - In INBOX or regular folders: "keep" = no action (leave in place)
            - In junk/spam folders: "keep" = rescue to INBOX (move email back)
            
            This allows a single rule to serve both purposes:
            1. Prevent important emails from being auto-filed (INBOX processing)
            2. Rescue wrongly-junked emails (Junk folder processing)
            
            See config/classification_rules.yaml for junk_folders configuration.
        """
        try:
            # Handle "keep" action - context-aware based on source folder
            if action.type == 'keep':
                if self._is_junk_folder():
                    # In junk folder: "keep" means rescue to INBOX
                    logger.info(f"🛟 Rescuing email from {self._current_folder} to INBOX (keep action in junk folder)")
                    action = EmailAction(type='move', folder='INBOX')
                    # Fall through to move handling below
                else:
                    # In regular folder: "keep" means no action needed
                    return True, None
            
            if action.type == 'move' and action.folder:
                # Check if already in destination folder (no-op)
                if action.folder == self._current_folder:
                    logger.debug(f"Skipping move: already in {action.folder}")
                    return True, f"ALREADY_IN:{action.folder}"
                
                # Check for cross-account move
                if (action.target_account and 
                    action.target_account != self.account_id):
                    
                    # Cross-account move requested but flag not enabled
                    if not self.allow_cross_account_moves:
                        logger.warning(f"Cross-account move skipped: Rule specifies target_account='{action.target_account}' "
                                     f"but --allow-cross-account-moves not set. Email stays in INBOX.")
                        return False, None  # Skip move - keep in INBOX rather than move to wrong account
                    
                    # Cross-account move enabled - proceed
                    if not self.cross_account_service:
                        logger.error("Cross-account move requested but CrossAccountMoveService not available")
                        return False, None
                    
                    # Check account permissions
                    if not self.account_manager.can_move_between(self.account_id, action.target_account):
                        logger.warning(f"Cross-account move blocked: {self.account_id} → {action.target_account} not allowed")
                        return False, None
                    
                    # Get email details from DB
                    if not self.use_database:
                        logger.error("Cross-account moves require database")
                        return False
                    
                    # get_db imported at module level
                    from backend.core.database.models import Email, CrossAccountMove
                    from backend.core.database.repository import EmailRepository
                    
                    db_session = next(get_db())
                    repo = EmailRepository(db_session)
                    
                    # Find email by UID and account
                    db_email = db_session.query(Email).filter(
                        Email.uid == uid,
                        Email.account_id == self.account_id
                    ).first()
                    
                    if not db_email:
                        logger.error(f"Email {uid} not found in database for account {self.account_id}")
                        db_session.close()
                        return False
                    
                    # Check for concurrent move (race condition prevention)
                    existing_move = db_session.query(CrossAccountMove).filter(
                        CrossAccountMove.message_id == db_email.message_id,
                        CrossAccountMove.status.in_(['pending', 'retrying', 'in_progress'])
                    ).first()
                    
                    if existing_move:
                        logger.warning(f"Move already in progress for {db_email.message_id} "
                                     f"(status: {existing_move.status}, initiated by: {existing_move.initiated_by})")
                        db_session.close()
                        return False  # Skip this move, let the existing one complete
                    
                    # Create move record (don't commit yet - wait for move result)
                    move_record = CrossAccountMove(
                        email_id=db_email.id,
                        message_id=db_email.message_id or '',
                        from_account_id=self.account_id,
                        from_folder=db_email.folder or 'INBOX',
                        to_account_id=action.target_account,
                        to_folder=action.folder,
                        move_method='pending',
                        status='pending',
                        initiated_by='rule',
                        rule_name=getattr(action, 'rule_name', None)
                    )
                    db_session.add(move_record)
                    # Don't commit yet - wait for move to succeed
                    
                    # Perform the move
                    success, error, move_method = await self.cross_account_service.move_email(
                        email_uid=uid,
                        message_id=db_email.message_id or '',
                        from_account=self.account_id,
                        from_folder=db_email.folder or 'INBOX',
                        to_account=action.target_account,
                        to_folder=action.folder,
                        from_imap=imap
                    )
                    
                    # Update move record and commit only if move succeeded
                    if success:
                        move_record.status = 'completed'
                        move_record.completed_at = datetime.utcnow()
                        move_record.move_method = move_method or 'unknown'
                        
                        # Update email record location
                        repo.track_location_change(
                            db_email,
                            new_folder=action.folder,
                            new_account=action.target_account,
                            moved_by='cross_account_rule',
                            move_reason=f"Rule: {getattr(action, 'rule_name', 'unknown')}"
                        )
                        
                        self.stats['actions_taken']['cross_account_move'] = self.stats['actions_taken'].get('cross_account_move', 0) + 1
                        logger.info(f"✅ Cross-account move completed: {self.account_id} → {action.target_account}")
                        
                        # Format cross-account info for display
                        cross_account_info = f"{self.account_id} → {action.target_account}: {action.folder}"
                        
                        # Commit transaction only after successful move
                        db_session.commit()
                        db_session.close()
                        
                        return True, cross_account_info
                    else:
                        move_record.status = 'failed'
                        move_record.error_message = error
                        move_record.retry_count = 0
                        move_record.next_retry_at = datetime.utcnow()  # Will be calculated by retry manager
                        logger.error(f"❌ Cross-account move failed: {error}")
                        
                        # Commit failed move record for retry tracking
                        db_session.commit()
                        db_session.close()
                        
                        return False, None
                
                # Regular same-account move
                # SECURITY: Double-check AI-generated actions use whitelisted folders
                if getattr(action, '_ai_generated', False):
                    if action.folder not in self.ai_allowed_folders:
                        logger.error(f"SECURITY VIOLATION: AI attempted to move to non-whitelisted folder '{action.folder}' - BLOCKING")
                        return False
                
                # Determine if folder creation is allowed:
                # 1. If --create-folders flag is set: always allow
                # 2. Otherwise: only allow for rule-based actions (not AI-generated)
                if self.create_folders:
                    create_if_missing = True
                else:
                    # Safety: AI-generated actions cannot auto-create folders unless --create-folders is set
                    create_if_missing = not getattr(action, '_ai_generated', False)
                
                success = imap.move_to_folder(uid, action.folder, create_if_missing=create_if_missing)
                if success:
                    self.stats['actions_taken']['move'] += 1
                    self.stats['folders_used'][action.folder] += 1
                    
                    # Update folder in database if DB is enabled
                    if self.use_database:
                        try:
                            from backend.core.database.models import Email
                            from backend.core.database.repository import EmailRepository
                            
                            db_session = next(get_db())
                            repo = EmailRepository(db_session)
                            
                            # Find email by UID and account
                            db_email = db_session.query(Email).filter(
                                Email.uid == uid,
                                Email.account_id == self.account_id
                            ).first()
                            
                            if db_email:
                                # Determine who moved it
                                moved_by = 'ai' if getattr(action, '_ai_generated', False) else 'rule'
                                rule_name = getattr(action, 'rule_name', None)
                                move_reason = f"Rule: {rule_name}" if rule_name else f"AI classification"
                                
                                repo.track_location_change(
                                    db_email,
                                    new_folder=action.folder,
                                    new_account=self.account_id,
                                    moved_by=moved_by,
                                    move_reason=move_reason
                                )
                                db_session.commit()
                                logger.debug(f"Updated DB folder: {uid} → {action.folder}")
                            
                            db_session.close()
                        except Exception as e:
                            logger.warning(f"Failed to update folder in DB: {e}")
                
                return success, None
            
            elif action.type == 'color' and action.color:
                success = imap.apply_color_label(uid, action.color)
                if success:
                    self.stats['actions_taken']['color'] += 1
                    self.stats['colors_applied'][action.color] += 1
                return success, None
            
            elif action.type == 'label' and action.labels:
                # Add custom IMAP flags
                for label in action.labels:
                    imap.add_custom_flag(uid, label)
                self.stats['actions_taken']['label'] += 1
                return True, None
            
            elif action.type == 'archive':
                # Archive = move to Archive folder
                # Archive is a rule-based action, so allow folder creation
                success = imap.move_to_folder(uid, 'Archive', create_if_missing=True)
                if success:
                    self.stats['actions_taken']['archive'] += 1
                    self.stats['folders_used']['Archive'] += 1
                    
                    # Update folder in database if DB is enabled
                    if self.use_database:
                        try:
                            from backend.core.database.models import Email
                            from backend.core.database.repository import EmailRepository
                            
                            db_session = next(get_db())
                            repo = EmailRepository(db_session)
                            
                            db_email = db_session.query(Email).filter(
                                Email.uid == uid,
                                Email.account_id == self.account_id
                            ).first()
                            
                            if db_email:
                                repo.track_location_change(
                                    db_email,
                                    new_folder='Archive',
                                    new_account=self.account_id,
                                    moved_by='rule',
                                    move_reason='Archive action'
                                )
                                db_session.commit()
                                logger.debug(f"Updated DB folder: {uid} → Archive")
                            
                            db_session.close()
                        except Exception as e:
                            logger.warning(f"Failed to update folder in DB: {e}")
                
                return success, None
            
            elif action.type == 'keep':
                # Keep in inbox (no action)
                return True, None
            
            elif action.type == 'forward' and action.forward_to:
                # Forward to another address (Phase 2)
                # Would use SMTP here
                self.stats['actions_taken']['forward'] += 1
                return True, None
            
            return False, None
            
        except Exception as e:
            self.stats['errors'].append({
                'uid': uid,
                'action': action.type,
                'error': str(e)
            })
            return False, None
    
    async def _process_batch_parallel(self, batch: List, imap, folder: str = "INBOX") -> None:
        """Process a batch of emails in parallel."""
        # Temporarily raise log level to WARNING to reduce interleaved noise
        original_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        
        tasks = []
        for uid, raw_email in batch:
            async def process_one(u, re, f):
                try:
                    processed = await self.processor.process(re, uid=u, folder=f)
                    result = await self.process_email(processed, imap)
                    return (u, result, None)
                except Exception as e:
                    return (u, None, e)
            tasks.append(process_one(uid, raw_email, folder))
        
        # Execute all in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Restore original log level
        logging.getLogger().setLevel(original_level)
        
        # Process results and print clean batch summary
        print()  # New line after progress message
        successful = []
        failed = []
        
        for item in results:
            if isinstance(item, Exception):
                failed.append(('unknown', str(item)))
                continue
                
            uid, result, error = item
            if error:
                self.stats['errors'].append({'uid': uid, 'error': str(error), 'type': type(error).__name__})
                failed.append((uid, str(error)))
            elif result:
                self.stats['processed'] += 1
                successful.append(result)
        
        # Print clean summary for this batch
        if successful:
            # Get cost stats if available
            cost_str = ""
            if self.ai_classifier and hasattr(self.ai_classifier, 'get_usage_stats'):
                stats = self.ai_classifier.get_usage_stats()
                total_cost = stats.get('total_cost_usd', 0)
                cost_str = f" (cost so far: ${total_cost:.4f})"
            print(f"  ✅ Batch completed: {len(successful)} emails processed{cost_str}")
            for result in successful:
                if result:  # Check if result is not None
                    ai_cat = result.get('ai_classification', {})
                    if ai_cat:
                        category = ai_cat.get('category', 'unknown')
                        confidence = ai_cat.get('confidence', 0.0)
                    else:
                        category = 'unknown'
                        confidence = 0.0
                    actions = result.get('actions', [])
                    stage_2 = ' [S2]' if any('Stage 2' in str(a) for a in actions) else ''
                    cross_account = ' 🔄' if result.get('cross_account_move') else ''
                    subject = result.get('subject', 'No Subject')[:45]
                    print(f"     [{result.get('uid', '?')}] {category:30} ({confidence:.2f}){stage_2:5}{cross_account} {subject}")
                    if result.get('cross_account_move'):
                        print(f"         🔄 Cross-account: {result['cross_account_move']}")
        
        if failed:
            print(f"  ❌ Errors: {len(failed)}")
            for uid, error in failed:
                print(f"     [{uid}] {error[:80]}")
    
    async def process_inbox(self,
                           imap_config: IMAPConfig,
                           folder: str = 'INBOX',
                           limit: Optional[int] = None,
                           new_only: bool = False,
                           since_date: Optional[str] = None,
                           full_sync: bool = False,
                           reset_sync_state: bool = False) -> Dict:
        """
        Process emails in inbox.
        
        Args:
            imap_config: IMAP connection configuration
            folder: Folder to process
            limit: Maximum number of emails to process
            new_only: If True, only process UNSEEN emails
            
        Returns:
            Processing statistics
        """
        print(f"\n{'='*80}")
        print(f"📧 EMAIL PROCESSING PIPELINE")
        print(f"{'='*80}")
        print(f"Account:  {self.account_id}")
        print(f"Folder:   {folder}")
        print(f"Mode:     {'New emails only (UNSEEN)' if new_only else 'All emails'}")
        if limit:
            print(f"Limit:    {limit} emails")
        print(f"Dry Run:  {'Yes' if self.dry_run else 'No'}")
        print(f"{'='*80}\n")
        
        start_time = datetime.now()
        
        # Store imap_config for use in process_email (e.g., for inquiry drafts)
        self._current_imap_config = imap_config
        
        # Store current folder for context-aware "keep" behavior
        # When processing junk folders, "keep" will rescue emails to INBOX
        self._current_folder = folder
        
        with IMAPMonitor(imap_config, timeout=self.imap_timeout, safe_move=self.safe_move) as imap:
            print(f"✅ Connected to {imap_config.host} (timeout: {self.imap_timeout}s)")
            if self.safe_move:
                print(f"🔒 Safe-move enabled: Post-copy verification active")
            
            # Get folders
            folders = imap.get_folder_list()
            print(f"📁 Available folders: {len(folders)}")
            
            # Check incremental sync state (if database enabled)
            last_uid = None
            sync_state = None
            if self.use_database and not full_sync and not new_only:
                try:
                    # get_db imported at module level
                    db_session = next(get_db())
                    repo = EmailRepository(db_session)
                    
                    if reset_sync_state:
                        # Reset sync state for this folder
                        print(f"🔄 Resetting sync state for {self.account_id}:{folder}...")
                        sync_state = repo.get_folder_sync_state(folder, account_id=self.account_id)
                        if sync_state:
                            db_session.delete(sync_state)
                            db_session.commit()
                            print(f"   ✓ Sync state reset")
                        sync_state = None
                    else:
                        sync_state = repo.get_folder_sync_state(folder, account_id=self.account_id)
                        if sync_state:
                            last_uid = sync_state.last_processed_uid
                            print(f"📊 Incremental sync: Last processed UID = {last_uid}")
                            print(f"   Last sync: {sync_state.last_sync_at}, Total processed: {sync_state.total_processed}")
                        else:
                            print(f"📊 First sync for {self.account_id}:{folder} (no previous state)")
                    
                    db_session.close()
                except Exception as e:
                    logger.warning(f"Could not load sync state: {e}")
                    last_uid = None
            
            # Special handling for --message-id filter: find and process just one email
            if hasattr(self, 'filter_message_id') and self.filter_message_id:
                print(f"\n🔍 Searching for email by Message-ID: {self.filter_message_id}")
                
                # Search for the email by Message-ID
                found_uid = None
                found_folder = None
                search_msg_id = self.filter_message_id.strip('<>') 
                
                # First try the specified folder, then search all folders
                folders_to_search = [folder] if folder != 'INBOX' else []
                folders_to_search.append('INBOX')
                # Add common folders
                for common_folder in ['MD/Applications/PhD', 'MD/Applications/Postdoc', 'MD/Applications/BScMScThesis', 'Archive']:
                    if common_folder not in folders_to_search and common_folder in folders:
                        folders_to_search.append(common_folder)
                
                for search_folder in folders_to_search:
                    try:
                        imap.client.select_folder(search_folder)
                        # Try different Message-ID formats
                        for msg_id_format in [search_msg_id, f'<{search_msg_id}>', self.filter_message_id]:
                            results = imap.client.search(['HEADER', 'MESSAGE-ID', msg_id_format])
                            if results:
                                found_uid = str(results[0])
                                found_folder = search_folder
                                print(f"   ✅ Found in '{search_folder}' with UID {found_uid}")
                                break
                        if found_uid:
                            break
                    except Exception as e:
                        logger.debug(f"Search in {search_folder} failed: {e}")
                        continue
                
                if not found_uid:
                    print(f"   ❌ Email not found with Message-ID: {self.filter_message_id}")
                    return
                
                # Update current folder context
                self._current_folder = found_folder
                
                # Fetch and process just this one email
                imap.client.select_folder(found_folder)
                fetch_data = imap.client.fetch([int(found_uid)], ['BODY.PEEK[]'])
                
                if int(found_uid) not in fetch_data:
                    print(f"   ❌ Could not fetch email with UID {found_uid}")
                    return
                
                raw_email = fetch_data[int(found_uid)].get(b'BODY[]', b'')
                if not raw_email:
                    print(f"   ❌ Email body is empty for UID {found_uid}")
                    return
                
                processed = await self.processor.process(raw_email, uid=found_uid, folder=found_folder)
                
                # Show email header
                self._print_email_card_header(1, "/1", processed)
                
                # Process the email
                result = await self.process_email(processed, imap)
                
                # Show result
                if result:
                    self._print_email_card_result(result)
                
                print(f"\n{'='*80}")
                print(f"✅ Processed 1 email by Message-ID")
                return
            
            # Fetch emails
            print(f"\n🔄 Fetching emails from {folder}...")
            
            email_count = 0
            highest_uid = last_uid or 0  # Track highest UID we process
            highest_successful_uid = last_uid or 0  # Track highest UID that was FULLY successful (no action failures)
            failed_uids = []  # Track UIDs that had failures (for safer sync state)
            emails_fetched_count = 0  # Track how many emails were fetched (for sync state logic)
            
            # OPTIMIZATION: For selective processing (embeddings-only, ai-only, etc),
            # first get all message IDs and check DB, then only fetch emails that need processing.
            # This is 100x faster than fetching every email just to check if it's processed.
            use_bulk_filtering = (
                self.use_database and 
                not new_only and 
                (self.generate_embeddings or self.use_ai)  # Only optimize for DB-backed operations
            )
            
            uids_to_process = None  # Will store filtered list if using bulk filtering
            
            if use_bulk_filtering:
                print(f"🚀 OPTIMIZATION: Bulk filtering enabled (checking DB before fetching email content)")
                
                try:
                    # Step 1: Get all message IDs from IMAP (fast - headers only)
                    message_inventory = imap.get_message_ids_with_headers(
                        folder=folder,
                        limit=limit,
                        since_date=since_date,
                        since_uid=last_uid if last_uid and not since_date else None
                    )
                    
                    if message_inventory:
                        print(f"   ✓ Retrieved {len(message_inventory)} message IDs from IMAP")
                        
                        # Step 2: Bulk check which ones need processing (single DB query)
                        # Add timeout handling for large inboxes
                        db_session = None
                        try:
                            db_session = next(get_db())
                            repo = EmailRepository(db_session)
                            from backend.core.database.models import Email
                        except Exception as e:
                            logger.error(f"Failed to get database session: {e}")
                            raise
                        
                        # Get unique message IDs (some emails may share the same Message-ID)
                        # Normalize message IDs consistently (bulk_check_processing_status normalizes them)
                        def normalize_msg_id(msg_id: str) -> str:
                            """Normalize message ID: ensure angle brackets"""
                            if not msg_id:
                                return msg_id
                            if not msg_id.startswith('<'):
                                return f'<{msg_id}>'
                            return msg_id
                        
                        # Filter out invalid message IDs (None, empty, or error messages)
                        valid_inventory = []
                        invalid_count = 0
                        for item in message_inventory:
                            msg_id = item.get('message_id')
                            if not msg_id or msg_id.startswith('<error-'):
                                invalid_count += 1
                                logger.debug(f"Skipping email UID {item.get('uid')}: invalid message_id '{msg_id}'")
                                continue
                            valid_inventory.append(item)
                        
                        if invalid_count > 0:
                            logger.warning(f"   Skipped {invalid_count} emails with invalid/missing message IDs")
                        
                        # Normalize all message IDs before deduplication to ensure consistent matching
                        normalized_inventory = [
                            {**item, 'message_id': normalize_msg_id(item['message_id'])}
                            for item in valid_inventory
                        ]
                        unique_message_ids = list(set(item['message_id'] for item in normalized_inventory if item['message_id']))
                        
                        logger.info(f"   Normalized {len(message_inventory)} message IDs to {len(unique_message_ids)} unique IDs")
                        
                        # Bulk check with error handling
                        try:
                            status_map = repo.bulk_check_processing_status(
                                unique_message_ids,
                                check_embeddings=self.generate_embeddings,
                                check_ai_classification=self.use_ai and not self.reprocess,
                                account_id=self.account_id
                            )
                        except Exception as e:
                            logger.error(f"Bulk status check failed: {e}")
                            # Fall back to processing all emails (safer than skipping)
                            logger.warning("Falling back to processing all emails (DB check failed)")
                            status_map = {}
                            # Mark all as needing processing
                            for msg_id in unique_message_ids:
                                status_map[msg_id] = {'in_db': False}
                        
                        # Verify status_map has entries for all unique message IDs
                        missing_in_status = len(unique_message_ids) - len(status_map)
                        if missing_in_status > 0:
                            logger.warning(f"   Warning: {missing_in_status} unique message IDs missing from status_map (normalization mismatch?)")
                        
                        # Debug: Check if emails exist in DB but with different account_id
                        found_count = sum(1 for s in status_map.values() if s.get('in_db'))
                        if found_count < len(unique_message_ids) * 0.5:  # Less than 50% found
                            logger.warning(f"   Only {found_count}/{len(unique_message_ids)} message IDs found in DB for account '{self.account_id}'")
                            logger.warning(f"   This might indicate emails were stored with a different account_id")
                            # Sample a larger set of message IDs to check across all accounts
                            sample_size = min(50, len(unique_message_ids))
                            sample_ids = list(unique_message_ids[:sample_size])
                            sample_check = repo.db.query(Email.message_id, Email.account_id).filter(
                                Email.message_id.in_(sample_ids)
                            ).all()
                            if sample_check:
                                account_ids_found = set(acc_id for _, acc_id in sample_check)
                                account_id_counts = {}
                                for _, acc_id in sample_check:
                                    account_id_counts[acc_id] = account_id_counts.get(acc_id, 0) + 1
                                logger.warning(f"   Sample check ({len(sample_check)} emails): Found account_ids: {account_id_counts}")
                                logger.warning(f"   Current account_id: '{self.account_id}'")
                                logger.warning(f"   💡 TIP: If emails were stored with different account_id, use that account_id or reprocess")
                            else:
                                logger.warning(f"   Sample check found NO emails in DB (they may be genuinely new)")
                        
                        # Use normalized inventory for processing
                        message_inventory = normalized_inventory
                        
                        if db_session:
                            db_session.close()
                        
                        # Step 3: Filter to only emails that need processing
                        uids_to_process = []
                        skip_counts = {
                            'already_complete': 0,
                            'needs_embedding_only': 0,
                            'needs_ai_only': 0,
                            'needs_both': 0,
                            'not_in_db': 0,
                            'not_in_status': 0
                        }
                        skip_reasons = {
                            'has_embedding': 0,
                            'has_ai': 0
                        }
                        
                        for item in normalized_inventory:
                            msg_id = item['message_id']
                            uid = item['uid']
                            status = status_map.get(msg_id)
                            
                            # If message_id not in status_map, need to process it (new email)
                            # This can happen if the message_id wasn't in the unique list or normalization mismatch
                            if status is None:
                                skip_counts['not_in_status'] += 1
                                uids_to_process.append(uid)  # Process to create email + embedding
                                continue
                            
                            needs_processing = False
                            
                            # If not in DB at all, need to process it
                            if not status.get('in_db'):
                                skip_counts['not_in_db'] += 1
                                needs_processing = True
                            else:
                                # Email is in DB - check what needs processing
                                needs_embedding = self.generate_embeddings and not status.get('has_embedding')
                                needs_ai = self.use_ai and (self.reprocess or not status.get('has_ai_classification'))
                                
                                if needs_embedding or needs_ai:
                                    needs_processing = True
                                    # Track granular reasons
                                    if needs_embedding and needs_ai:
                                        skip_counts['needs_both'] += 1
                                    elif needs_embedding:
                                        skip_counts['needs_embedding_only'] += 1
                                    elif needs_ai:
                                        skip_counts['needs_ai_only'] += 1
                                else:
                                    # Email is complete - has everything needed
                                    skip_counts['already_complete'] += 1
                                
                                # Track what we're skipping (for reporting)
                                if self.generate_embeddings and status.get('has_embedding'):
                                    skip_reasons['has_embedding'] += 1
                                if self.use_ai and status.get('has_ai_classification') and not self.reprocess:
                                    skip_reasons['has_ai'] += 1
                            
                            if needs_processing:
                                uids_to_process.append(uid)
                        
                        skipped = len(message_inventory) - len(uids_to_process)
                        emails_fetched_count = len(message_inventory)  # Track fetched count
                        print(f"   ✓ Filtered to {len(uids_to_process)} emails needing processing ({skipped} skipped)")
                        
                        # Build detailed skip breakdown
                        skip_parts = []
                        if skip_counts['already_complete'] > 0:
                            skip_parts.append(f"{skip_counts['already_complete']} already complete")
                        if skip_counts['needs_embedding_only'] > 0:
                            skip_parts.append(f"{skip_counts['needs_embedding_only']} need embeddings only")
                        if skip_counts['needs_ai_only'] > 0:
                            skip_parts.append(f"{skip_counts['needs_ai_only']} need AI classification only")
                        if skip_counts['needs_both'] > 0:
                            skip_parts.append(f"{skip_counts['needs_both']} need both embeddings and AI")
                        if skip_counts['not_in_db'] > 0:
                            skip_parts.append(f"{skip_counts['not_in_db']} not in DB")
                        if skip_counts['not_in_status'] > 0:
                            skip_parts.append(f"{skip_counts['not_in_status']} no status (new emails)")
                        
                        if skip_parts:
                            print(f"      Skipped: {', '.join(skip_parts)}")
                        
                        # Store skip counts and fetched count for later reporting
                        self.stats['skip_counts'] = skip_counts
                        self.stats['fetched'] = emails_fetched_count
                        
                        # Track highest UID from inventory for sync state (even if no emails to process)
                        if message_inventory:
                            highest_uid = max(int(item['uid']) for item in message_inventory)
                        elif last_uid:
                            # No new emails, but we should still update sync state with last_uid
                            highest_uid = last_uid
                    else:
                        print(f"   No emails found in folder")
                        uids_to_process = []
                        # Still track UID for sync state if we have a last_uid
                        if last_uid:
                            highest_uid = last_uid
                        
                except Exception as e:
                    logger.warning(f"Bulk filtering failed, falling back to normal mode: {e}")
                    use_bulk_filtering = False
                    uids_to_process = None
            
            # LIFECYCLE-ONLY MODE: Super lightweight - just check message IDs and update folders
            # Don't fetch email content at all - use BULK updates
            if not self.use_ai and not self.generate_embeddings and not self.classifier and not self.vip_manager:
                print(f"🔄 LIFECYCLE-ONLY: Bulk folder tracking (no email content fetching)")
                
                # Get message IDs without fetching email content
                message_inventory = imap.get_message_ids_with_headers(
                    folder=folder,
                    limit=limit,
                    since_date=since_date,
                    since_uid=last_uid if last_uid and not since_date else None
                )
                
                if message_inventory:
                    print(f"   Found {len(message_inventory)} emails in {folder}")
                    
                    # Prepare data for bulk update
                    message_data = [(item['message_id'], item['uid']) for item in message_inventory]
                    
                    # BULK update folders (single transaction, very fast)
                    db_session = next(get_db())
                    repo = EmailRepository(db_session)
                    
                    stats = repo.bulk_update_folder_lifecycle(message_data, folder, account_id=self.account_id)
                    
                    db_session.commit()
                    db_session.close()
                    
                    print(f"   ✓ Lifecycle tracking complete:")
                    print(f"      {stats['updated']} folder changes detected and tracked")
                    print(f"      {stats['unchanged']} emails already in {folder}")
                    print(f"      {stats['not_in_db']} emails not in DB (skipped)")
                    
                    self.stats['processed'] = stats['updated']
                    
                    # Track highest UID
                    if message_inventory:
                        highest_uid = max(int(item['uid']) for item in message_inventory)
                
                # Skip to end (no email processing needed)
                email_iter = iter([])  # Empty iterator
                
            # Choose fetch method
            elif new_only:
                email_iter = imap.fetch_unseen_emails(limit=limit)
                print(f"   Mode: UNSEEN emails only")
            elif use_bulk_filtering and uids_to_process is not None and len(uids_to_process) > 0:
                # Fetch only the filtered UIDs (already have the list)
                def fetch_filtered_emails():
                    """Generator that fetches only the UIDs we need"""
                    for uid in uids_to_process:
                        try:
                            msg_id = int(uid)
                            fetch_data = imap.client.fetch([msg_id], ['BODY.PEEK[]'])
                            if msg_id in fetch_data:
                                raw_email = fetch_data[msg_id].get(b'BODY[]', fetch_data[msg_id].get(b'RFC822'))
                                yield (uid, raw_email)
                        except Exception as e:
                            logger.error(f"Error fetching UID {uid}: {e}")
                
                email_iter = fetch_filtered_emails()
                print(f"   Mode: Bulk filtered - fetching {len(uids_to_process)} emails that need processing")
            elif use_bulk_filtering and uids_to_process is not None and len(uids_to_process) == 0:
                # All emails already processed - create empty iterator
                print(f"   Mode: All emails already processed (no new emails to fetch)")
                email_iter = iter([])
            else:
                # Use incremental sync if we have a last_uid
                if last_uid and not since_date:
                    # Fetch only emails after last_uid
                    email_iter = imap.fetch_all_emails(folder=folder, limit=limit, since_uid=last_uid)
                    print(f"   Mode: Incremental (UIDs > {last_uid}, most recent first)")
                else:
                    email_iter = imap.fetch_all_emails(folder=folder, limit=limit, since_date=since_date)
                    if since_date:
                        print(f"   Mode: All emails since {since_date} (most recent first)")
                    elif full_sync:
                        print(f"   Mode: Full sync - all emails (most recent first)")
                    else:
                        print(f"   Mode: All emails (most recent first)")
            
            # Track total emails for card output
            total_emails_to_process = None
            if use_bulk_filtering and uids_to_process is not None:
                total_emails_to_process = len(uids_to_process)
            elif not use_bulk_filtering:
                # Will be determined as we process
                total_emails_to_process = None
            
            # Process in parallel batches (streaming, don't load all into memory)
            if self.parallel_workers > 1:
                batch = []
                batch_num = 0
                
                for uid, raw_email in email_iter:
                    # Track highest UID for incremental sync
                    uid_int = int(uid)
                    if uid_int > highest_uid:
                        highest_uid = uid_int
                    
                    batch.append((uid, raw_email))
                    
                    # Process when batch is full
                    if len(batch) >= self.parallel_workers:
                        batch_num += 1
                        email_count += len(batch)
                        emails_fetched_count = max(emails_fetched_count, email_count)  # Track fetched count
                        
                        # Show progress with embedding stats
                        emb_gen = self.stats.get('embeddings_generated', 0)
                        emb_skip = self.stats.get('embeddings_skipped', 0)
                        progress_msg = f"  Processing batch {batch_num} ({len(batch)} emails in parallel)..."
                        if self.generate_embeddings:
                            progress_msg += f" | Embeddings: {emb_gen} gen, {emb_skip} skip"
                        print(progress_msg, end='\r')
                        
                        # Process this batch in parallel
                        await self._process_batch_parallel(batch, imap, folder)
                        
                        # Keep IMAP connection alive after batch processing
                        try:
                            imap.keep_alive()
                        except Exception:
                            pass  # Reconnection will happen on next fetch if needed
                        
                        # Clear batch for next iteration
                        batch = []
                
                # Process remaining emails in final batch (if any)
                if batch:
                    batch_num += 1
                    email_count += len(batch)
                    emails_fetched_count = max(emails_fetched_count, email_count)  # Track fetched count
                    emb_gen = self.stats.get('embeddings_generated', 0)
                    emb_skip = self.stats.get('embeddings_skipped', 0)
                    progress_msg = f"  Processing final batch {batch_num} ({len(batch)} emails)..."
                    if self.generate_embeddings:
                        progress_msg += f" | Embeddings: {emb_gen} gen, {emb_skip} skip"
                    print(progress_msg, end='\r')
                    await self._process_batch_parallel(batch, imap, folder)
                    
                    # Keep IMAP connection alive after final batch
                    try:
                        imap.keep_alive()
                    except Exception:
                        pass
                    
            else:
                # Sequential processing (original code)
                for uid, raw_email in email_iter:
                    email_count += 1
                    emails_fetched_count = max(emails_fetched_count, email_count)  # Track fetched count
                    
                    # Progress messages suppressed - cards show progress instead
                    
                    try:
                        # Track highest UID for incremental sync
                        uid_int = int(uid)
                        if uid_int > highest_uid:
                            highest_uid = uid_int
                        
                        # Process email
                        processed = await self.processor.process(raw_email, uid=uid, folder=folder)
                        
                        # Apply date filters (ensure timezone-aware comparison)
                        if self.filter_date_after and processed.date:
                            # Make date timezone-aware if it's naive
                            email_date = processed.date
                            if email_date.tzinfo is None:
                                email_date = email_date.replace(tzinfo=timezone.utc)
                            if email_date < self.filter_date_after:
                                continue  # Skip emails before cutoff date
                        
                        if self.filter_date_before and processed.date:
                            # Make date timezone-aware if it's naive
                            email_date = processed.date
                            if email_date.tzinfo is None:
                                email_date = email_date.replace(tzinfo=timezone.utc)
                            if email_date > self.filter_date_before:
                                continue  # Skip emails after cutoff date
                        
                        # Show email header immediately (before processing)
                        total_display = f"/{total_emails_to_process}" if total_emails_to_process else ""
                        self._print_email_card_header(email_count, total_display, processed)
                        
                        # Run through pipeline
                        result = await self.process_email(processed, imap)
                        
                        self.stats['processed'] += 1
                        
                        # Check if processing was fully successful (no action failures)
                        # Action failures include: move timeout, flag timeout, etc.
                        has_action_failure = result.get('action_error') or result.get('move_failed')
                        if has_action_failure:
                            failed_uids.append(uid_int)
                            logger.warning(f"UID {uid} had action failure, will not update sync state past this point")
                        elif uid_int > highest_successful_uid and not failed_uids:
                            # Only update highest_successful_uid if no earlier failures
                            highest_successful_uid = uid_int
                        
                        # Show AI classification and action (after processing)
                        self._print_email_card_result(result)
                        
                        # Print cost summary every 5 emails (only in sequential mode)
                        if email_count % 5 == 0 and self.parallel_workers == 1:
                            self._print_cost_summary_batch(email_count)
                        
                        # Keep IMAP connection alive after processing each email
                        # This prevents timeout during long AI classification
                        try:
                            imap.keep_alive()
                        except Exception:
                            pass  # Reconnection will happen on next fetch if needed
                        
                    except Exception as e:
                        self.stats['errors'].append({
                            'uid': uid,
                            'error': str(e),
                            'type': type(e).__name__
                        })
                        # Print error in the result section
                        error_result = {
                            'uid': uid,
                            'error': str(e)
                        }
                        self._print_email_card_result(error_result)
            
            print(f"\n\n✅ Processing complete")
            
            # Update sync state ONLY if we processed everything (no limit hit)
            # This ensures we don't skip emails on the next run
            # Also update if no emails found (to record that we checked up to highest_uid)
            # For --limit: Update if all fetched emails were handled (processed or skipped as complete)
            emails_fetched = self.stats.get('fetched', max(emails_fetched_count, email_count))
            emails_processed = self.stats.get('processed', 0)
            skip_counts = self.stats.get('skip_counts', {})
            emails_handled = emails_processed + skip_counts.get('already_complete', 0)
            
            # IMPORTANT: If any emails had action failures (move/flag timeout), use safe UID
            # This ensures failed emails are retried on the next run
            if failed_uids:
                safe_sync_uid = highest_successful_uid
                logger.warning(f"Using safe sync UID {safe_sync_uid} due to {len(failed_uids)} failed emails: {failed_uids[:5]}{'...' if len(failed_uids) > 5 else ''}")
            else:
                safe_sync_uid = highest_uid
            
            # Update sync state if:
            # 1. No limit set (processed all available), OR
            # 2. Limit set but all fetched emails were handled (processed or skipped as complete)
            should_update_sync = (
                self.use_database and 
                safe_sync_uid >= (last_uid or 0) and  # >= instead of > to allow update when no new emails
                (not limit or (limit and emails_fetched > 0 and emails_handled >= emails_fetched))
            )
            
            if should_update_sync:
                try:
                    # get_db imported at module level
                    db_session = next(get_db())
                    repo = EmailRepository(db_session)
                    repo.update_folder_sync_state(folder, safe_sync_uid, emails_processed, account_id=self.account_id)
                    db_session.commit()
                    db_session.close()
                    if failed_uids:
                        print(f"⚠️  Updated sync state: {self.account_id}:{folder} → UID {safe_sync_uid} ({len(failed_uids)} failed emails will be retried)")
                    elif emails_processed > 0:
                        print(f"📊 Updated sync state: {self.account_id}:{folder} → UID {safe_sync_uid} ({emails_processed} processed)")
                    elif emails_handled > 0:
                        print(f"📊 Updated sync state: {self.account_id}:{folder} → UID {safe_sync_uid} ({emails_handled} handled, all complete)")
                    else:
                        print(f"📊 Updated sync state: {self.account_id}:{folder} → UID {safe_sync_uid} (no new emails found)")
                except Exception as e:
                    logger.warning(f"Failed to update sync state: {e}")
            elif limit and highest_uid > (last_uid or 0):
                if emails_handled < emails_fetched:
                    print(f"⚠️  Sync state NOT updated (limit={limit} hit, {emails_fetched - emails_handled} emails still pending). Run without --limit to update sync state.")
                else:
                    print(f"⚠️  Sync state NOT updated (limit={limit} hit, but all fetched emails handled). Consider running without --limit to update sync state.")
        
        # Generate report
        elapsed = (datetime.now() - start_time).total_seconds()
        self._print_report(elapsed)
        
        # Print cost summary
        self._print_cost_summary()
        
        return self.stats
    
    def _print_email_card_header(self, email_num: int, total_display: str, email: ProcessedEmail):
        """
        Print the email card header (before processing - shows immediately).
        
        Shows:
        - Email number, UID, date
        - Sender and subject
        
        Args:
            email_num: Current email number (1-indexed)
            total_display: String like "/74" or "" if total unknown
            email: ProcessedEmail object
        """
        # Format date
        if email.date:
            if isinstance(email.date, datetime):
                date_str = email.date.strftime('%Y-%m-%d %H:%M')
            else:
                date_str = str(email.date)
        else:
            date_str = 'Unknown'
        
        # Format sender
        from_name = email.from_name or ''
        from_addr = email.from_address or 'Unknown'
        if from_name:
            sender_display = f"{from_name} <{from_addr}>"
        else:
            sender_display = from_addr
        
        # Truncate long sender/subject
        max_sender_len = 50
        max_subject_len = 60
        if len(sender_display) > max_sender_len:
            sender_display = sender_display[:max_sender_len-3] + '...'
        subject = email.subject or 'No Subject'
        if len(subject) > max_subject_len:
            subject = subject[:max_subject_len-3] + '...'
        
        # Print card header
        print(f"\n{'━' * 80}", flush=True)
        print(f"📬 Email {email_num}{total_display} • UID {email.uid} • {date_str}", flush=True)
        print(flush=True)
        print(f"From:     {sender_display}", flush=True)
        print(f"Subject:  {subject}", flush=True)
    
    def _print_email_card_result(self, result: Dict):
        """
        Print the email card result (after processing - AI and actions).
        
        Shows:
        - Preprocessing (if changed)
        - VIP level
        - Rule match
        - AI classification
        - Extra details for VIP/urgent/errors/invitations
        - Actions taken
        
        Args:
            result: Processing result dictionary
        """
        # Determine if we should show extra detail
        ai_cat = result.get('ai_classification', {}) if isinstance(result.get('ai_classification'), dict) else {}
        category = ai_cat.get('category', '')
        is_vip = result.get('vip_level') is not None
        is_urgent = result.get('rule_matched') and 'urgent' in str(result.get('rule_matched', '')).lower()
        is_error = result.get('error') is not None
        is_invitation = category.startswith('invitation-')
        show_extra_detail = is_vip or is_urgent or is_error or is_invitation
        
        # Show preprocessing only if it changed something
        if result.get('preprocessing_changed'):
            print(f"          🔄 Preprocessed: {result.get('original_from')} → {result.get('from')}", flush=True)
        
        # Show VIP level
        if is_vip:
            vip_level = result.get('vip_level', '').upper()
            vip_emoji = {'URGENT': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡'}
            emoji = vip_emoji.get(vip_level, '⚪')
            print(f"          {emoji} VIP: {vip_level}", flush=True)
        
        # Show rule match
        if result.get('rule_matched'):
            rule_desc = str(result.get('rule_matched'))
            if 'urgent' in rule_desc.lower():
                print(f"          ⚠️  Rule: {rule_desc}", flush=True)
            else:
                print(f"          📋 Rule: {rule_desc}", flush=True)
        
        # Show AI classification
        if ai_cat:
            category = ai_cat.get('category', '')
            confidence = ai_cat.get('confidence', 0.0)
            ai_time = result.get('ai_time', 0)
            
            ai_line = f"AI:       {category} (confidence: {confidence:.2f}"
            if ai_time > 0:
                ai_line += f", {ai_time:.1f}s"
            ai_line += ")"
            print(ai_line, flush=True)
        
        # Show extra detail for VIP, urgent, errors, invitations
        if show_extra_detail:
            print()
            
            # Error details
            if is_error:
                print(f"          ❌ ERROR: {result.get('error')}")
            
            # Invitation details
            if is_invitation and ai_cat:
                deadline = ai_cat.get('deadline')
                event_date = ai_cat.get('event_date')
                location = ai_cat.get('location')
                time_commitment = ai_cat.get('time_commitment_hours')
                relevance = ai_cat.get('relevance_score')
                prestige = ai_cat.get('prestige_score')
                
                if deadline:
                    print(f"          📅 Deadline: {deadline}")
                if event_date:
                    location_str = f" ({location})" if location else ""
                    print(f"          📅 Event: {event_date}{location_str}")
                if time_commitment:
                    print(f"          ⏰ Time commitment: {time_commitment} hours")
                if relevance is not None:
                    print(f"          📊 Relevance: {relevance}/10")
                if prestige is not None:
                    print(f"          ⭐ Prestige: {prestige}/10")
            
            # Application details (if available)
            if category.startswith('application-') and ai_cat:
                applicant_name = ai_cat.get('applicant_name')
                applicant_institution = ai_cat.get('applicant_institution')
                scientific_excellence = ai_cat.get('scientific_excellence_score')
                recommendation = ai_cat.get('recommendation_score')
                
                if applicant_name:
                    inst_str = f" ({applicant_institution})" if applicant_institution else ""
                    print(f"          👤 Applicant: {applicant_name}{inst_str}")
                if scientific_excellence is not None:
                    print(f"          📊 Scientific excellence: {scientific_excellence}/10")
                if recommendation is not None:
                    print(f"          ✅ Recommendation: {recommendation}/10")
            
            # Urgency details
            if is_urgent or (ai_cat and ai_cat.get('urgency_score', 0) >= 8):
                urgency_score = ai_cat.get('urgency_score') if ai_cat else None
                if urgency_score:
                    print(f"          ⚠️  Urgency score: {urgency_score}/10")
                deadline = ai_cat.get('deadline') if ai_cat else None
                if deadline:
                    print(f"          📅 Deadline: {deadline}")
        
        # Show actions
        actions = result.get('executed', [])
        dry_run_actions = result.get('dry_run_actions', [])
        
        if actions:
            print(flush=True)
            for i, action in enumerate(actions):
                if i == 0:
                    print(f"Action:   ", end='', flush=True)
                else:
                    print(f"          ", end='', flush=True)
                # Format action nicely
                if 'Cross-account move' in action:
                    print(f"🔄 {action}", flush=True)
                elif 'Applied AI action: move' in action:
                    # Extract category from "Applied AI action: move (category)"
                    if '(' in action and ')' in action:
                        category = action.split('(')[1].split(')')[0]
                        print(f"✓ Moved → AI: {category}", flush=True)
                    else:
                        print(f"✓ {action}", flush=True)
                elif 'Applied rule action: move' in action:
                    print(f"✓ Moved (rule match)", flush=True)
                elif 'Moved' in action or '→' in action:
                    # Extract folder from action string
                    if '→' in action:
                        folder = action.split('→')[-1].strip()
                        print(f"✓ Moved → {folder}", flush=True)
                    else:
                        print(f"✓ {action}", flush=True)
                elif 'flag' in action.lower() or '$' in action:
                    print(f"✓ {action}", flush=True)
                else:
                    print(f"✓ {action}", flush=True)
        elif dry_run_actions:
            print(flush=True)
            for i, action in enumerate(dry_run_actions):
                if i == 0:
                    print(f"Action:   ", end='', flush=True)
                else:
                    print(f"          ", end='', flush=True)
                print(f"🔍 DRY RUN: {action}", flush=True)
        elif result.get('cross_account_move'):
            print(flush=True)
            print(f"Action:   🔄 Cross-account move: {result['cross_account_move']}", flush=True)
    
    def _print_cost_summary_batch(self, email_count: int):
        """Print cost summary every N emails (called every 5 emails in sequential mode)"""
        total_cost = 0.0
        
        # AI Classification costs
        if self.ai_classifier:
            ai_stats = self.ai_classifier.get_usage_stats()
            total_cost += ai_stats.get('total_cost_usd', 0.0)
        
        # Embedding costs
        if self.embedding_generator:
            emb_stats = self.embedding_generator.get_usage_stats()
            total_cost += emb_stats.get('total_cost_usd', 0.0)
        
        if total_cost > 0:
            cost_per_email = total_cost / email_count if email_count > 0 else 0
            print(f"\n{'━' * 80}")
            print(f"📊 Batch 1-{email_count} processed • 💰 Cost: ${total_cost:.4f} (avg: ${cost_per_email:.4f}/email)")
            print(f"{'━' * 80}")
    
    def _print_cost_summary(self):
        """Print OpenAI API cost summary"""
        print(f"\n{'='*70}")
        print(f"💰 API COST SUMMARY")
        print(f"{'='*70}")
        
        total_cost = 0.0
        
        # AI Classification costs
        if self.ai_classifier:
            ai_stats = self.ai_classifier.get_usage_stats()
            print(f"\n🤖 AI Classification ({ai_stats['model']}):")
            print(f"   Tokens: {ai_stats['total_tokens']:,} "
                  f"(prompt: {ai_stats['total_prompt_tokens']:,}, "
                  f"completion: {ai_stats['total_completion_tokens']:,})")
            print(f"   Cost: ${ai_stats['total_cost_usd']:.4f}")
            total_cost += ai_stats['total_cost_usd']
        
        # Embedding costs
        if self.embedding_generator:
            emb_stats = self.embedding_generator.get_usage_stats()
            print(f"\n🔍 Vector Embeddings ({emb_stats['model']}):")
            print(f"   Tokens: {emb_stats['total_tokens']:,}")
            print(f"   Cost: ${emb_stats['total_cost_usd']:.4f}")
            total_cost += emb_stats['total_cost_usd']
        
        # Total
        print(f"\n{'─'*70}")
        print(f"💵 TOTAL API COST: ${total_cost:.4f}")
        
        # Projection
        if self.stats['processed'] > 0:
            cost_per_email = total_cost / self.stats['processed']
            print(f"\n📊 Cost Breakdown:")
            print(f"   Per email: ${cost_per_email:.4f}")
            print(f"   Per 100 emails: ${cost_per_email * 100:.2f}")
            print(f"   Per 1000 emails: ${cost_per_email * 1000:.2f}")
            print(f"   Per 10,000 emails: ${cost_per_email * 10000:.2f}")
        
        print(f"{'='*70}\n")
    
    def _print_report(self, elapsed: float):
        """Print processing statistics"""
        print(f"\n{'='*70}")
        print(f"📊 PROCESSING REPORT")
        print(f"{'='*70}")
        
        total = self.stats['processed']
        print(f"Total Emails Processed: {total}")
        print(f"Processing Time: {elapsed:.1f}s ({total/elapsed if elapsed > 0 else 0:.1f} emails/s)")
        print(f"")
        
        # VIP stats
        if self.stats['vip_detected'] > 0:
            print(f"🔴 VIP Emails Detected: {self.stats['vip_detected']} ({self.stats['vip_detected']/total*100:.1f}%)")
        
        # Rule matching
        if self.stats['rule_matched'] > 0:
            print(f"📋 Emails Matched Rules: {self.stats['rule_matched']} ({self.stats['rule_matched']/total*100:.1f}%)")
        
        # AI classification stats (Phase 2)
        if self.stats['ai_classified'] > 0 or self.stats['ai_failures'] > 0:
            print(f"🤖 AI Classifications: {self.stats['ai_classified']}")
            if self.stats['ai_reprocessed'] > 0:
                print(f"   🔄 Reprocessed: {self.stats['ai_reprocessed']}")
            if self.stats['ai_failures'] > 0:
                print(f"   ⚠️  AI Failures: {self.stats['ai_failures']}")
        
        # Database stats (Phase 2)
        if self.stats['db_stored'] > 0 or self.stats['db_failures'] > 0:
            print(f"💾 Database Stored: {self.stats['db_stored']}")
            if self.stats['db_failures'] > 0:
                print(f"   ⚠️  DB Failures: {self.stats['db_failures']}")
        
        # Phase 3 stats
        if self.stats['embeddings_generated'] > 0 or self.stats['embeddings_skipped'] > 0:
            print(f"🔍 Embeddings Generated: {self.stats['embeddings_generated']}")
            if self.stats['embeddings_skipped'] > 0:
                print(f"   ⏭️  Skipped (existing): {self.stats['embeddings_skipped']}")
            if self.stats['embeddings_failures'] > 0:
                print(f"   ⚠️  Failures: {self.stats['embeddings_failures']}")
        
        if self.stats['responses_tracked'] > 0:
            print(f"📬 Responses Tracked: {self.stats['responses_tracked']}")
        
        if self.stats['drafts_generated'] > 0:
            print(f"✍️  Drafts Generated: {self.stats['drafts_generated']}")
            if self.stats['drafts_failures'] > 0:
                print(f"   ⚠️  Failures: {self.stats['drafts_failures']}")
        
        # Actions taken
        if sum(self.stats['actions_taken'].values()) > 0:
            print(f"\n⚡ ACTIONS EXECUTED:")
            for action_type, count in self.stats['actions_taken'].most_common():
                print(f"   {action_type}: {count}")
            
            # Folder breakdown
            if self.stats['folders_used']:
                print(f"\n📁 Folders Used:")
                for folder, count in self.stats['folders_used'].most_common():
                    print(f"   {count:4d} → {folder}")
            
            # Color breakdown
            if self.stats['colors_applied']:
                print(f"\n🎨 Colors Applied:")
                color_names = {1: "Red", 2: "Orange", 3: "Yellow", 5: "Blue", 6: "Purple", 7: "Gray"}
                for color, count in sorted(self.stats['colors_applied'].items()):
                    print(f"   {count:4d} → {color_names.get(color, f'Color {color}')}")
        
        # Errors
        if self.stats['errors']:
            print(f"\n❌ Errors: {len(self.stats['errors'])}")
            for i, err in enumerate(self.stats['errors'][:5], 1):
                print(f"   {i}. UID {err['uid']}: {err['error'][:80]}")
        
        # Dry run notice
        if self.dry_run:
            print(f"\n🔍 DRY RUN: No changes were made to the IMAP server")
            print(f"   Run without --dry-run to actually apply changes")
        
        print(f"{'='*70}\n")


async def main():
    """Main entry point"""
    # Environment already loaded at module level
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Process IMAP inbox with rules and AI classification'
    )
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would happen without making changes')
    parser.add_argument('--new-only', action='store_true',
                       help='Process only UNSEEN emails')
    parser.add_argument('--limit', type=int, default=None,
                       help='Maximum number of emails to process')
    parser.add_argument('--all', action='store_true',
                       help='Process ALL emails (overrides --limit)')
    parser.add_argument('--limit-past', type=int, default=None, metavar='DAYS',
                       help='Only process emails from the last N days (e.g., --limit-past 100)')
    parser.add_argument('--folder', type=str, default='INBOX',
                       help='Folder to process (default: INBOX)')
    parser.add_argument('--recursive', action='store_true',
                       help='Process folder and all subfolders recursively (e.g., --folder INBOX --recursive processes INBOX, INBOX/Subfolder, etc.)')
    parser.add_argument('--skip-vip', action='store_true',
                       help='Skip VIP detection')
    parser.add_argument('--skip-rules', action='store_true',
                       help='Skip rule-based classification')
    parser.add_argument('--skip-ai', action='store_true',
                       help='Skip AI classification (Phase 2)')
    parser.add_argument('--use-two-stage', action='store_true',
                       help='Use two-stage classifier (gpt-5-mini → gpt-5.1) instead of one-stage (gpt-5-mini)')
    parser.add_argument('--skip-database', action='store_true',
                       help='Skip database storage (Phase 2)')
    parser.add_argument('--reprocess', action='store_true',
                       help='Rerun AI classification on all emails, even if previously processed')
    parser.add_argument('--skip-embeddings', action='store_true',
                       help='Skip vector embedding generation (Phase 3, default: ON)')
    parser.add_argument('--skip-tracking', action='store_true',
                       help='Skip response tracking (Phase 3, default: ON)')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip preprocessing (forwarded email detection)')
    parser.add_argument('--skip-actions', action='store_true',
                       help='Skip action execution (like dry-run but still processes)')
    parser.add_argument('--generate-drafts', action='store_true',
                       help='Generate draft replies using AI answer_options (Phase 3, default: OFF)')

    # Attachment backfill mode
    parser.add_argument('--backfill-attachments', action='store_true',
                       help='Backfill attachment indexing for existing emails (fetches from IMAP)')
    parser.add_argument('--backfill-since', type=str, default=None, metavar='DATE',
                       help='Only backfill emails since this date (YYYY-MM-DD format)')
    parser.add_argument('--backfill-until', type=str, default=None, metavar='DATE',
                       help='Only backfill emails until this date (YYYY-MM-DD format)')
    parser.add_argument('--backfill-retry-failed', action='store_true',
                       help='Include previously failed emails in backfill (respects backoff)')
    parser.add_argument('--backfill-stats-only', action='store_true',
                       help='Only show backfill statistics, do not process')

    parser.add_argument(
        '--safe-move',
        action='store_true',
        default=True,
        dest='safe_move',
        help='Enable safe moving with post-copy verification and extra logging (DEFAULT: ON)'
    )
    parser.add_argument(
        '--no-safe-move',
        action='store_false',
        dest='safe_move',
        help='Disable safe-move verification (faster but riskier - emails may be lost if folder missing)'
    )
    parser.add_argument(
        '--create-folders',
        action='store_true',
        help='Auto-create IMAP folders if they don\'t exist (default: require pre-existing folders)'
    )
    
    # Exclusive modes (run ONLY this stage)
    parser.add_argument('--rules-only', action='store_true',
                       help='Run ONLY rule-based classification (skip AI, embeddings, tracking)')
    parser.add_argument('--ai-only', action='store_true',
                       help='Run ONLY AI classification (skip rules, VIP, actions)')
    parser.add_argument('--embeddings-only', action='store_true',
                       help='Run ONLY embedding generation (requires --use-database)')
    parser.add_argument('--tracking-only', action='store_true',
                       help='Run ONLY response tracking analysis')
    parser.add_argument('--lifecycle-only', action='store_true',
                       help='Run ONLY lifecycle tracking (detect folder changes, NO AI/rules/embeddings). Use for Archive/Trash scanning.')
    parser.add_argument('--actions-only', action='store_true',
                       help='Execute actions ONLY on already-processed emails (skip AI, embeddings, tracking). Uses existing ai_category from database.')
    
    # Filtering options
    parser.add_argument('--message-id', type=str,
                       help='Process only the email with this specific Message-ID (e.g., "<abc123@example.com>")')
    parser.add_argument('--filter-category', type=str,
                       help='Process only emails in this category (e.g., application-phd)')
    parser.add_argument('--filter-unclassified', action='store_true',
                       help='Process only emails without AI classification')
    parser.add_argument('--filter-no-embedding', action='store_true',
                       help='Process only emails without embeddings')
    parser.add_argument('--filter-needs-reply', action='store_true',
                       help='Process only emails marked as needing replies')
    parser.add_argument('--filter-vip-only', action='store_true',
                       help='Process only VIP emails')
    parser.add_argument('--filter-date-after', type=str,
                       help='Process only emails after date (YYYY-MM-DD)')
    parser.add_argument('--filter-date-before', type=str,
                       help='Process only emails before date (YYYY-MM-DD)')
    
    # Force operations
    parser.add_argument('--force-reembed', action='store_true',
                       help='Regenerate embeddings even if they exist')
    parser.add_argument('--force-retrack', action='store_true',
                       help='Re-analyze reply needs even if already tracked')
    
    # Incremental sync
    parser.add_argument('--full-sync', action='store_true',
                       help='Process all emails (default: incremental - only new emails since last sync)')
    parser.add_argument('--reset-sync-state', action='store_true',
                       help='Reset sync state for folder (start from beginning)')
    parser.add_argument('--set-sync-to-latest', action='store_true',
                       help='Set sync state to latest UID without processing (skip old emails)')
    
    # Performance options
    parser.add_argument('--parallel-workers', type=int, default=1,
                       help='Number of parallel workers for processing (1-50, default: 1). '
                            'Each worker processes emails concurrently. Recommended: 10-30 for bulk operations.')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed processing info')
    
    # Multi-account support
    parser.add_argument('--account', type=str, default=None,
                       help='Account to process (configured in config/accounts.yaml). '
                            'If not specified, uses default from config (typically "work")')
    parser.add_argument('--all-accounts', action='store_true',
                       help='Process all configured accounts in sequence')
    parser.add_argument('--list-accounts', action='store_true',
                       help='List configured accounts and exit')
    parser.add_argument('--test-connections', action='store_true',
                       help='Test IMAP and SMTP connections for all configured accounts')
    parser.add_argument('--allow-cross-account-moves', action='store_true',
                       help='Allow rules to move emails between accounts (CAUTION!)')
    parser.add_argument('--run-orphan-cleanup', action='store_true',
                       help='Run orphan cleanup service to detect duplicates and retry failed moves')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='Maximum retries for failed cross-account moves (default: 3)')
    
    # Inquiry handling options
    parser.add_argument('--list-drafts', action='store_true',
                       help='List pending inquiry drafts with Apple Mail links')
    parser.add_argument('--update-draft-lifecycle', action='store_true',
                       help='Update lifecycle status of pending inquiry drafts (sent/skipped detection)')
    
    args = parser.parse_args()
    
    # Handle --list-drafts (inquiry drafts)
    if args.list_drafts:
        try:
            from backend.core.database import init_db, get_db
            from backend.core.email.inquiry_lifecycle import get_pending_drafts, format_pending_drafts_report
            
            init_db()
            db_session = next(get_db())
            
            drafts = get_pending_drafts(db_session)
            report = format_pending_drafts_report(drafts)
            print(report)
            
            db_session.close()
            sys.exit(0)
            
        except Exception as e:
            print(f"❌ Error listing drafts: {e}")
            sys.exit(1)
    
    # Handle --update-draft-lifecycle
    if args.update_draft_lifecycle:
        try:
            from backend.core.database import init_db, get_db
            from backend.core.email.inquiry_lifecycle import update_draft_lifecycle, get_pending_drafts, format_pending_drafts_report
            from backend.core.email.imap_drafts import IMAPDraftsManager
            
            init_db()
            db_session = next(get_db())
            imap_drafts = IMAPDraftsManager()
            
            print("\n📋 Updating inquiry draft lifecycle status...")
            result = update_draft_lifecycle(db_session, imap_drafts)
            
            print(f"\n✅ Lifecycle check complete:")
            print(f"   Checked: {result['checked']}")
            print(f"   Sent: {result['sent']}")
            print(f"   Skipped: {result['skipped']}")
            print(f"   Still pending: {result['still_pending']}")
            if result.get('errors'):
                print(f"   Errors: {result['errors']}")
            
            # Show remaining pending drafts
            drafts = get_pending_drafts(db_session)
            if drafts:
                report = format_pending_drafts_report(drafts)
                print(report)
            
            db_session.close()
            sys.exit(0)
            
        except Exception as e:
            print(f"❌ Error updating draft lifecycle: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Handle --list-accounts
    if args.list_accounts:
        try:
            from backend.core.accounts.manager import AccountManager
            account_manager = AccountManager()
            
            print("\n" + "="*70)
            print("CONFIGURED EMAIL ACCOUNTS")
            print("="*70)
            
            for nickname in account_manager.list_accounts():
                info = account_manager.get_account_display_info(nickname)
                default_marker = " (default)" if nickname == account_manager.default_account else ""
                print(f"\n{nickname}{default_marker}:")
                print(f"  Display Name: {info['display_name']}")
                print(f"  Email: {info['email']}")
                print(f"  IMAP Host: {info['host']}")
                if info['allow_moves_to']:
                    print(f"  Can move to: {', '.join(info['allow_moves_to'])}")
                else:
                    print(f"  Can move to: (none)")
            
            print(f"\n{'='*70}\n")
            sys.exit(0)
            
        except Exception as e:
            print(f"❌ Error loading accounts: {e}")
            print(f"\nCheck config/accounts.yaml and environment variables.")
            sys.exit(1)
    
    # Handle --test-connections
    if args.test_connections:
        try:
            from backend.core.accounts.manager import AccountManager
            from backend.core.email.imap_monitor import IMAPMonitor
            import smtplib
            
            account_manager = AccountManager()
            
            print("\n" + "="*70)
            print("TESTING ACCOUNT CONNECTIONS")
            print("="*70)
            
            results = {}
            
            for nickname in account_manager.list_accounts():
                account = account_manager.get_account(nickname)
                print(f"\n📧 Testing account: {nickname} ({account.display_name})")
                print(f"   Email: {account.imap_username or 'N/A'}")
                print(f"   IMAP: {account.imap_host}:{account.imap_port}")
                print(f"   SMTP: {account.smtp_host}:{account.smtp_port if account.smtp_host else 'N/A'}")
                
                account_results = {'imap': None, 'smtp': None}
                
                # Test IMAP connection
                print(f"\n   🔍 Testing IMAP connection...")
                try:
                    imap_config = account_manager.get_imap_config(nickname)
                    imap = IMAPMonitor(imap_config, timeout=120)  # Use longer timeout for testing
                    imap.connect()
                    
                    # Try to list folders as a real test
                    folders = imap.get_folder_list()
                    account_results['imap'] = {
                        'success': True,
                        'folder_count': len(folders),
                        'message': f"✅ Connected successfully ({len(folders)} folders available)"
                    }
                    print(f"      ✅ IMAP: Connected successfully ({len(folders)} folders available)")
                    
                    imap.disconnect()
                except Exception as e:
                    account_results['imap'] = {
                        'success': False,
                        'error': str(e),
                        'message': f"❌ IMAP: {str(e)}"
                    }
                    print(f"      ❌ IMAP: {str(e)}")
                
                # Test SMTP connection
                if account.smtp_host:
                    print(f"\n   📤 Testing SMTP connection...")
                    try:
                        # Create SMTP connection based on port
                        if account.smtp_port == 465:
                            # Port 465 uses SSL from the start
                            import ssl
                            context = ssl.create_default_context()
                            smtp = smtplib.SMTP_SSL(account.smtp_host, account.smtp_port, context=context, timeout=10)
                        else:
                            # Port 587 uses STARTTLS
                            smtp = smtplib.SMTP(account.smtp_host, account.smtp_port, timeout=10)
                            if account.smtp_use_tls:
                                smtp.starttls()
                        
                        # Try to login (but don't actually send anything)
                        if account.smtp_username and account.smtp_password:
                            smtp.login(account.smtp_username, account.smtp_password)
                            account_results['smtp'] = {
                                'success': True,
                                'message': f"✅ Connected and authenticated successfully"
                            }
                            print(f"      ✅ SMTP: Connected and authenticated successfully")
                        else:
                            account_results['smtp'] = {
                                'success': True,
                                'message': f"✅ Connected (no credentials to test)"
                            }
                            print(f"      ✅ SMTP: Connected (no credentials provided)")
                        
                        smtp.quit()
                    except Exception as e:
                        # Sanitize error message to prevent password leaks
                        error_msg = str(e)
                        if account.smtp_password and account.smtp_password in error_msg:
                            error_msg = error_msg.replace(account.smtp_password, "***")
                        account_results['smtp'] = {
                            'success': False,
                            'error': error_msg,
                            'message': f"❌ SMTP: {error_msg}"
                        }
                        print(f"      ❌ SMTP: {error_msg}")
                else:
                    account_results['smtp'] = {
                        'success': None,
                        'message': "⏭️  SMTP: Not configured"
                    }
                    print(f"      ⏭️  SMTP: Not configured")
                
                results[nickname] = account_results
            
            # Print summary
            print(f"\n{'='*70}")
            print("CONNECTION TEST SUMMARY")
            print("="*70)
            
            all_imap_ok = True
            all_smtp_ok = True
            
            for nickname, result in results.items():
                account = account_manager.get_account(nickname)
                print(f"\n{nickname}:")
                
                if result['imap']:
                    if result['imap']['success']:
                        print(f"  ✅ IMAP: {result['imap']['message']}")
                    else:
                        print(f"  ❌ IMAP: {result['imap']['message']}")
                        all_imap_ok = False
                else:
                    print(f"  ⚠️  IMAP: Not tested")
                
                if result['smtp']:
                    if result['smtp']['success'] is True:
                        print(f"  ✅ SMTP: {result['smtp']['message']}")
                    elif result['smtp']['success'] is False:
                        print(f"  ❌ SMTP: {result['smtp']['message']}")
                        all_smtp_ok = False
                    else:
                        print(f"  ⏭️  SMTP: {result['smtp']['message']}")
                else:
                    print(f"  ⚠️  SMTP: Not tested")
            
            # Check if any SMTP accounts are configured
            has_smtp_accounts = any(
                account_manager.get_account(nickname).smtp_host 
                for nickname in results.keys()
            )
            
            print(f"\n{'='*70}")
            if all_imap_ok and (not has_smtp_accounts or all_smtp_ok):
                print("✅ All connections successful!")
                exit_code = 0
            else:
                print("⚠️  Some connections failed. Check credentials and network.")
                exit_code = 1
            print(f"{'='*70}\n")
            
            sys.exit(exit_code)
            
        except Exception as e:
            print(f"❌ Error testing connections: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Handle exclusive modes (override skip flags)
    if args.rules_only:
        args.skip_ai = True
        args.skip_embeddings = True
        args.skip_tracking = True
        args.generate_drafts = False
        logger.info("📋 RULES-ONLY MODE: Running preprocessing + VIP + rules + actions")
    
    elif args.ai_only:
        args.skip_vip = True
        args.skip_rules = True
        args.skip_embeddings = True
        args.skip_actions = True
        logger.info("🤖 AI-ONLY MODE: Running AI classification + database only")
    
    elif args.embeddings_only:
        args.skip_vip = True
        args.skip_rules = True
        args.skip_ai = True
        args.skip_tracking = True
        args.skip_actions = True
        args.skip_database = False  # MUST have database for embeddings
        logger.info("🔍 EMBEDDINGS-ONLY MODE: Generating vector embeddings only")
    
    elif args.tracking_only:
        args.skip_vip = True
        args.skip_rules = True
        args.skip_ai = True  # Use existing AI data
        args.skip_embeddings = True
        args.skip_actions = True
        args.skip_database = False  # MUST have database for tracking
        logger.info("📊 TRACKING-ONLY MODE: Analyzing reply needs only")
    
    elif args.lifecycle_only:
        args.skip_vip = True
        args.skip_rules = True
        args.skip_ai = True
        args.skip_embeddings = True
        args.skip_tracking = True
        args.skip_actions = True
        args.skip_database = False  # MUST have database for lifecycle tracking
        logger.info("🔄 LIFECYCLE-ONLY MODE: Tracking folder changes only (NO AI/rules/embeddings)")
    
    # Handle --set-sync-to-latest (special mode that doesn't process emails)
    if args.set_sync_to_latest:
        print(f"\n{'='*70}")
        print(f"📊 SETTING SYNC STATE TO LATEST UID")
        print(f"{'='*70}\n")
        print(f"Folder: {args.folder}")
        
        # Connect to IMAP to get latest UID
        try:
            from backend.core.database import init_db, get_db
            from backend.core.database.repository import EmailRepository
            
            init_db()
            
            host = os.getenv('IMAP_HOST')
            port = int(os.getenv('IMAP_PORT', 993))
            username = os.getenv('IMAP_USERNAME')
            password = os.getenv('IMAP_PASSWORD')
            
            config = IMAPConfig(host=host, username=username, password=password, port=port, folder=args.folder)
            imap_timeout = int(os.getenv('IMAP_TIMEOUT', '120'))
            
            with IMAPMonitor(config, timeout=imap_timeout) as imap:
                print(f"✅ Connected to {host} (timeout: {imap_timeout}s)")
                imap.client.select_folder(args.folder)
                # Get all message UIDs
                all_messages = imap.client.search(['ALL'])
                if all_messages:
                    latest_uid = max(all_messages)
                    print(f"📊 Latest UID in {args.folder}: {latest_uid}")
                    
                    # Update sync state (use default account)
                    db_session = next(get_db())
                    repo = EmailRepository(db_session)
                    account_id = "work"  # Default for legacy mode
                    repo.update_folder_sync_state(args.folder, latest_uid, emails_processed=0, account_id=account_id)
                    db_session.commit()
                    db_session.close()
                    
                    print(f"✅ Sync state set: {account_id}:{args.folder} → UID {latest_uid}")
                    print(f"   Next run will only process emails newer than UID {latest_uid}")
                else:
                    print(f"⚠️  No emails found in {args.folder}")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        sys.exit(0)
    
    # Handle --run-orphan-cleanup
    if args.run_orphan_cleanup:
        try:
            from backend.core.accounts.manager import AccountManager
            from backend.core.email.orphan_cleanup import OrphanCleanupService
            from backend.core.database import get_db
            
            account_manager = AccountManager()
            db_session = next(get_db())
            
            cleanup_service = OrphanCleanupService(
                db_session, account_manager, dry_run=args.dry_run
            )
            
            import asyncio
            stats = asyncio.run(cleanup_service.run_cleanup())
            
            print(f"\n{'='*70}")
            print(f"ORPHAN CLEANUP RESULTS")
            print(f"{'='*70}")
            print(f"Duplicates detected: {stats['duplicates_detected']}")
            print(f"Duplicates removed: {stats['duplicates_removed']}")
            print(f"Failed moves retried: {stats['failed_moves_retried']}")
            print(f"Orphans cleaned: {stats['orphans_cleaned']}")
            print(f"{'='*70}\n")
            
            db_session.close()
            sys.exit(0)

        except ImportError:
            print("❌ Error: OrphanCleanupService not yet implemented (Phase 3)")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error running orphan cleanup: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Handle --backfill-attachments
    if args.backfill_attachments:
        try:
            from datetime import datetime as dt
            from backend.core.accounts.manager import AccountManager
            from backend.core.database import init_db, get_db
            from backend.core.database.repository import EmailRepository
            from backend.core.email.attachment_extractor import AttachmentExtractor, EmailNotFoundError, IMAPConnectionError
            from backend.core.documents.attachment_indexer import AttachmentIndexer
            from backend.core.email.models import AttachmentInfo

            account_manager = AccountManager()
            init_db()
            db_session = next(get_db())
            repo = EmailRepository(db_session)

            # Parse dates
            since_date = None
            until_date = None
            if args.backfill_since:
                since_date = dt.strptime(args.backfill_since, '%Y-%m-%d')
            if args.backfill_until:
                until_date = dt.strptime(args.backfill_until, '%Y-%m-%d')

            # Determine account filter
            account_filter = args.account if args.account else None

            print(f"\n{'='*70}")
            print(f"ATTACHMENT BACKFILL")
            print(f"{'='*70}")
            if since_date:
                print(f"Since: {since_date.strftime('%Y-%m-%d')}")
            if until_date:
                print(f"Until: {until_date.strftime('%Y-%m-%d')}")
            if account_filter:
                print(f"Account: {account_filter}")
            print(f"Include retries: {args.backfill_retry_failed}")
            print(f"Dry run: {args.dry_run}")
            print()

            # Get stats
            stats = repo.get_attachment_backfill_stats(
                since_date=since_date,
                account_id=account_filter
            )

            print(f"Emails with attachments: {stats['total_with_attachments']}")
            print(f"  Already indexed (success): {stats['indexed_success']}")
            print(f"  Already indexed (partial): {stats['indexed_partial']}")
            print(f"  Failed (will retry): {stats['failed']}")
            print(f"  Pending: {stats['pending']}")
            print(f"  Never attempted: {stats['never_attempted']}")
            print()

            if args.backfill_stats_only:
                db_session.close()
                sys.exit(0)

            # Get emails to process
            limit = args.limit if args.limit else 100
            emails = repo.get_emails_for_attachment_backfill(
                since_date=since_date,
                until_date=until_date,
                limit=limit,
                include_retries=args.backfill_retry_failed,
                account_id=account_filter,
            )

            if not emails:
                print("No emails need attachment indexing.")
                db_session.close()
                sys.exit(0)

            print(f"Processing {len(emails)} emails...\n")

            # Initialize attachment extractor
            extractor = AttachmentExtractor(account_manager)

            # Process each email
            processed = 0
            success = 0
            failed = 0
            skipped = 0

            for idx, email_obj in enumerate(emails, 1):
                try:
                    account_id = email_obj.account_id or 'work'
                    prefix = f"[{idx}/{len(emails)}]"

                    # Get attachment info from stored metadata
                    attachment_info_list = email_obj.attachment_info or []
                    if not attachment_info_list:
                        print(f"{prefix} {email_obj.from_address[:30]:30} - SKIP (no attachment info)")
                        repo.update_attachment_index_status(email_obj, 'success')  # Nothing to index
                        db_session.commit()
                        skipped += 1
                        continue

                    print(f"{prefix} {email_obj.from_address[:30]:30} ({email_obj.date.strftime('%Y-%m-%d')}) - {len(attachment_info_list)} attachment(s)", end='', flush=True)

                    if args.dry_run:
                        print(" [DRY RUN]")
                        continue

                    # Define callback to update email location if moved
                    def on_location_update(new_folder, new_uid):
                        email_obj.folder = new_folder
                        email_obj.uid = new_uid
                        db_session.commit()

                    # Fetch all attachments from IMAP
                    try:
                        attachments = extractor.get_all_attachments(
                            account_id=account_id,
                            folder=email_obj.folder,
                            uid=email_obj.uid,
                            message_id=email_obj.message_id,
                            on_location_update=on_location_update
                        )
                    except EmailNotFoundError as e:
                        print(f" - FAIL (email not found on IMAP)")
                        repo.update_attachment_index_status(
                            email_obj, 'failed', error=str(e), increment_attempts=True
                        )
                        db_session.commit()
                        failed += 1
                        continue
                    except IMAPConnectionError as e:
                        print(f" - FAIL (IMAP connection error)")
                        repo.update_attachment_index_status(
                            email_obj, 'failed', error=str(e), increment_attempts=True
                        )
                        db_session.commit()
                        failed += 1
                        continue

                    if not attachments:
                        print(f" - SKIP (no attachments found)")
                        repo.update_attachment_index_status(email_obj, 'success')
                        db_session.commit()
                        skipped += 1
                        continue

                    # Convert to AttachmentInfo objects for indexer
                    attachment_infos = []
                    attachment_contents = []
                    for content, filename, content_type in attachments:
                        attachment_infos.append(AttachmentInfo(
                            filename=filename,
                            content_type=content_type,
                            size=len(content),
                            extracted_text=None,
                            extraction_error=None
                        ))
                        attachment_contents.append(content)

                    # Index attachments
                    indexer = AttachmentIndexer(db_session)
                    import asyncio
                    results = asyncio.get_event_loop().run_until_complete(
                        indexer.index_email_attachments(
                            email=email_obj,
                            attachment_infos=attachment_infos,
                            attachment_contents=attachment_contents,
                        )
                    )

                    indexed_count = len(results)
                    new_count = sum(1 for _, is_new in results if is_new)

                    if indexed_count > 0:
                        print(f" - OK ({indexed_count} indexed, {new_count} new)")
                        repo.update_attachment_index_status(email_obj, 'success')
                        success += 1
                    elif len(attachments) > 0:
                        print(f" - PARTIAL (0/{len(attachments)} indexable)")
                        repo.update_attachment_index_status(email_obj, 'partial')
                        success += 1
                    else:
                        print(f" - SKIP (no indexable attachments)")
                        repo.update_attachment_index_status(email_obj, 'success')
                        skipped += 1

                    db_session.commit()
                    processed += 1

                except Exception as e:
                    print(f" - ERROR: {e}")
                    logger.error(f"Error processing email {email_obj.id}: {e}", exc_info=True)
                    repo.update_attachment_index_status(
                        email_obj, 'failed', error=str(e), increment_attempts=True
                    )
                    db_session.commit()
                    failed += 1

            print(f"\n{'='*70}")
            print(f"BACKFILL COMPLETE")
            print(f"{'='*70}")
            print(f"Processed: {processed}")
            print(f"Success: {success}")
            print(f"Failed: {failed}")
            print(f"Skipped: {skipped}")
            print(f"{'='*70}\n")

            db_session.close()
            sys.exit(0)

        except Exception as e:
            print(f"❌ Error during attachment backfill: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Handle --all-accounts
    if args.all_accounts:
        try:
            from backend.core.accounts.manager import AccountManager
            account_manager = AccountManager()
            
            results = {}
            print(f"\n{'='*70}")
            print(f"PROCESSING ALL ACCOUNTS")
            print(f"{'='*70}\n")
            
            for nickname in account_manager.list_accounts():
                print(f"\n>>> Processing account: {nickname}")
                print(f"{'─'*70}")
                
                # Create IMAP config for this account
                imap_config = account_manager.get_imap_config(nickname)
                
                # Create pipeline for this account
                pipeline = EmailProcessingPipeline(
                    account_id=nickname,
                    allow_cross_account_moves=args.allow_cross_account_moves,
                    preprocessing_rules=str(get_config_path("preprocessing_rules.yaml")) if not args.skip_preprocessing and not args.skip_rules and get_config_path("preprocessing_rules.yaml") else None,
                    classification_rules=str(get_config_path("classification_rules.yaml")) if not args.skip_rules and get_config_path("classification_rules.yaml") else None,
                    vip_config=str(get_config_path("vip_senders.yaml")) if not args.skip_vip and get_config_path("vip_senders.yaml") else None,
                    dry_run=args.dry_run or args.skip_actions,
                    use_database=not args.skip_database,
                    use_ai=not args.skip_ai,
                    use_two_stage=args.use_two_stage,
                    reprocess=args.reprocess,
                    generate_embeddings=not args.skip_embeddings,
                    track_responses=not args.skip_tracking,
                    generate_drafts=args.generate_drafts,
                    parallel_embeddings=args.parallel_workers,
                    safe_move=args.safe_move,
                    create_folders=args.create_folders,
                    actions_only=args.actions_only
                )
                
                # Process
                since_date = args.filter_date_after if hasattr(args, 'filter_date_after') and args.filter_date_after else None
                limit = None if args.all else args.limit
                
                # Handle recursive folder processing
                if args.recursive:
                    print(f"🔄 RECURSIVE MODE: Processing '{args.folder}' and all subfolders for account {nickname}\n")
                    
                    # Get all folders from IMAP server
                    # Import here to avoid scoping issues
                    from backend.core.email.imap_monitor import IMAPMonitor as IMAPMonitorClass
                    imap_timeout = int(os.getenv('IMAP_TIMEOUT', '120'))
                    with IMAPMonitorClass(imap_config, timeout=imap_timeout, safe_move=args.safe_move) as imap:
                        all_folders = imap.get_folder_list()
                    
                    # Filter folders that match the base folder path
                    base_folder = args.folder.rstrip('/')
                    matching_folders = []
                    
                    for folder in all_folders:
                        # Exact match or starts with base_folder/
                        if folder == base_folder or folder.startswith(base_folder + '/'):
                            matching_folders.append(folder)
                    
                    matching_folders.sort()  # Process in alphabetical order
                    
                    if not matching_folders:
                        print(f"❌ No folders found matching '{args.folder}' for account {nickname}")
                        results[nickname] = {'processed': 0, 'folders_processed': 0}
                        continue
                    
                    print(f"📁 Found {len(matching_folders)} folder(s) to process for {nickname}")
                    
                    # Process each folder
                    account_total_processed = 0
                    account_folders_processed = 0
                    
                    for folder in matching_folders:
                        try:
                            folder_stats = await pipeline.process_inbox(
                                imap_config,
                                folder=folder,
                                limit=limit,
                                new_only=args.new_only,
                                since_date=since_date,
                                full_sync=args.full_sync if hasattr(args, 'full_sync') else False,
                                reset_sync_state=args.reset_sync_state if hasattr(args, 'reset_sync_state') else False
                            )
                            
                            account_total_processed += folder_stats.get('processed', 0)
                            account_folders_processed += 1
                            
                        except Exception as e:
                            print(f"❌ Error processing {folder} for {nickname}: {e}")
                            logger.error(f"Error processing folder {folder} for account {nickname}: {e}", exc_info=True)
                            continue
                    
                    stats = {'processed': account_total_processed, 'folders_processed': account_folders_processed}
                else:
                    # Single folder processing (existing behavior)
                    stats = await pipeline.process_inbox(
                        imap_config,
                        folder=args.folder,
                        limit=limit,
                        new_only=args.new_only,
                        since_date=since_date,
                        full_sync=args.full_sync if hasattr(args, 'full_sync') else False,
                        reset_sync_state=args.reset_sync_state if hasattr(args, 'reset_sync_state') else False
                    )
                
                results[nickname] = stats
                print(f"\n✅ {nickname} complete: {stats['processed']} emails processed")
            
            # Print combined summary
            print(f"\n{'='*70}")
            print(f"ALL ACCOUNTS SUMMARY")
            print(f"{'='*70}")
            for nickname, stats in results.items():
                print(f"{nickname:15} {stats['processed']:5d} emails processed")
            print(f"{'='*70}\n")
            
            sys.exit(0)
            
        except Exception as e:
            print(f"❌ Error processing all accounts: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Validate exclusive mode requirements
    if args.embeddings_only or args.tracking_only:
        if args.skip_database:
            logger.error("❌ Error: --embeddings-only and --tracking-only require database")
            sys.exit(1)
    
    # Display filter info if any
    filters_active = []
    if hasattr(args, 'message_id') and args.message_id:
        filters_active.append(f"message-id={args.message_id}")
    if args.filter_category:
        filters_active.append(f"category={args.filter_category}")
    if args.filter_unclassified:
        filters_active.append("unclassified only")
    if args.filter_no_embedding:
        filters_active.append("no embeddings only")
    if args.filter_needs_reply:
        filters_active.append("needs reply only")
    if args.filter_vip_only:
        filters_active.append("VIP only")
    if args.filter_date_after:
        filters_active.append(f"after {args.filter_date_after}")
    if args.filter_date_before:
        filters_active.append(f"before {args.filter_date_before}")
    
    if filters_active:
        logger.info(f"🔍 FILTERS ACTIVE: {', '.join(filters_active)}")
    
    # Initialize AccountManager and get account to use
    try:
        from backend.core.accounts.manager import AccountManager
        account_manager = AccountManager()
        
        # Determine account to use
        account_id = args.account
        if not account_id:
            account_id = account_manager.default_account
        
        # Validate account exists
        account_config = account_manager.get_account(account_id)
        
        # Get IMAP config for selected account
        imap_config = account_manager.get_imap_config(account_id)
        
        print(f"📧 Email Account: {account_config.display_name} ({account_id})")
        print(f"🌐 IMAP Server: {account_config.imap_host}:{account_config.imap_port}")
        if args.actions_only:
            print(f"⚡ ACTIONS-ONLY MODE: Skipping AI/embeddings/tracking, executing actions based on existing classifications")
        
        # Warn about cross-account moves
        if args.allow_cross_account_moves:
            print(f"\n⚠️  WARNING: Cross-account moves are ENABLED!")
            print(f"   Emails may be moved to: {', '.join(account_config.allow_moves_to) if account_config.allow_moves_to else '(none)'}")
            if not args.dry_run:
                response = input("   Continue? (yes/no): ")
                if response.lower() != 'yes':
                    print("Cancelled.")
                    sys.exit(0)
        
    except ImportError:
        # Fallback to legacy mode if AccountManager not available
        logger.warning("AccountManager not available, using legacy single-account mode")
        account_id = "work"
        host = os.getenv('IMAP_HOST')
        port = int(os.getenv('IMAP_PORT', 993))
        username = os.getenv('IMAP_USERNAME')
        password = os.getenv('IMAP_PASSWORD')
        
        if not all([host, username, password]):
            print("❌ Error: Missing IMAP credentials in .env file")
            sys.exit(1)
        
        imap_config = IMAPConfig(
            host=host,
            username=username,
            password=password,
            port=port,
            folder=args.folder
        )
        
        print(f"📧 Email Account: {username} (legacy mode)")
        print(f"🌐 IMAP Server: {host}:{port}")
    except Exception as e:
        print(f"❌ Error loading account configuration: {e}")
        print(f"\nAvailable accounts:")
        try:
            for name in account_manager.list_accounts():
                print(f"  - {name}")
        except:
            pass
        sys.exit(1)
    
    # Determine limit
    limit = None if args.all else args.limit
    
    # Convert --limit-past to date filter
    if args.limit_past:
        from datetime import datetime, timedelta, timezone
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=args.limit_past)
        args.filter_date_after = cutoff_date.strftime('%Y-%m-%d')
        print(f"📅 Processing only emails from last {args.limit_past} days (since {args.filter_date_after})")
    
    # Confirm if processing many emails without dry-run
    if not args.dry_run:
        if args.all:
            response = input("\n⚠️  You're about to MODIFY ALL emails. This is PERMANENT. Continue? (yes/no): ")
            if response.lower() != 'yes':
                print("Cancelled.")
                sys.exit(0)
        elif limit and limit > 100:
            response = input(f"\n⚠️  You're about to modify {limit} emails. Continue? (yes/no): ")
            if response.lower() != 'yes':
                print("Cancelled.")
                sys.exit(0)
        elif args.new_only:
            response = input(f"\n⚠️  Process new emails (UNSEEN) and apply rules? (yes/no): ")
            if response.lower() != 'yes':
                print("Cancelled.")
                sys.exit(0)
    
    # Create pipeline (imap_config already created above via AccountManager or legacy mode)
    pipeline = EmailProcessingPipeline(
        account_id=account_id,
        allow_cross_account_moves=args.allow_cross_account_moves,
        preprocessing_rules=str(get_config_path("preprocessing_rules.yaml")) if not args.skip_preprocessing and not args.skip_rules and get_config_path("preprocessing_rules.yaml") else None,
        classification_rules=str(get_config_path("classification_rules.yaml")) if not args.skip_rules and get_config_path("classification_rules.yaml") else None,
        vip_config=str(get_config_path("vip_senders.yaml")) if not args.skip_vip and get_config_path("vip_senders.yaml") else None,
        dry_run=args.dry_run or args.skip_actions,
        use_database=not args.skip_database,
        use_ai=not args.skip_ai,
        use_two_stage=args.use_two_stage,
        reprocess=args.reprocess,
        generate_embeddings=not args.skip_embeddings,
        track_responses=not args.skip_tracking,
        generate_drafts=args.generate_drafts,
        parallel_embeddings=args.parallel_workers,
        safe_move=args.safe_move,
        create_folders=args.create_folders,
        actions_only=args.actions_only
    )
    
    # Add filtering attributes for email filtering
    pipeline.filter_message_id = args.message_id if hasattr(args, 'message_id') else None
    pipeline.filter_category = args.filter_category if hasattr(args, 'filter_category') else None
    pipeline.filter_unclassified = args.filter_unclassified if hasattr(args, 'filter_unclassified') else False
    pipeline.filter_no_embedding = args.filter_no_embedding if hasattr(args, 'filter_no_embedding') else False
    pipeline.filter_needs_reply = args.filter_needs_reply if hasattr(args, 'filter_needs_reply') else False
    pipeline.filter_vip_only = args.filter_vip_only if hasattr(args, 'filter_vip_only') else False
    # Parse dates as timezone-aware (UTC)
    from datetime import timezone
    if hasattr(args, 'filter_date_after') and args.filter_date_after:
        date_after = datetime.fromisoformat(args.filter_date_after)
        # Make timezone-aware if naive
        if date_after.tzinfo is None:
            date_after = date_after.replace(tzinfo=timezone.utc)
        pipeline.filter_date_after = date_after
    else:
        pipeline.filter_date_after = None
    
    if hasattr(args, 'filter_date_before') and args.filter_date_before:
        date_before = datetime.fromisoformat(args.filter_date_before)
        if date_before.tzinfo is None:
            date_before = date_before.replace(tzinfo=timezone.utc)
        pipeline.filter_date_before = date_before
    else:
        pipeline.filter_date_before = None
    pipeline.force_reembed = args.force_reembed if hasattr(args, 'force_reembed') else False
    pipeline.force_retrack = args.force_retrack if hasattr(args, 'force_retrack') else False
    
    # Process emails
    try:
        # Pass since_date if limit_past was specified
        since_date = args.filter_date_after if hasattr(args, 'filter_date_after') and args.filter_date_after else None
        
        # Handle recursive folder processing
        if args.recursive:
            print(f"🔄 RECURSIVE MODE: Processing '{args.folder}' and all subfolders\n")
            
            # Get all folders from IMAP server
            # Import here to avoid scoping issues
            from backend.core.email.imap_monitor import IMAPMonitor as IMAPMonitorClass
            imap_timeout = int(os.getenv('IMAP_TIMEOUT', '120'))
            with IMAPMonitorClass(imap_config, timeout=imap_timeout, safe_move=args.safe_move) as imap:
                all_folders = imap.get_folder_list()
            
            # Filter folders that match the base folder path
            base_folder = args.folder.rstrip('/')
            matching_folders = []
            
            for folder in all_folders:
                # Exact match or starts with base_folder/
                if folder == base_folder or folder.startswith(base_folder + '/'):
                    matching_folders.append(folder)
            
            matching_folders.sort()  # Process in alphabetical order
            
            if not matching_folders:
                print(f"❌ No folders found matching '{args.folder}'")
                sys.exit(1)
            
            print(f"📁 Found {len(matching_folders)} folder(s) to process:")
            for folder in matching_folders:
                print(f"   - {folder}")
            print()
            
            # Process each folder
            all_stats = []
            total_processed = 0
            total_errors = 0
            
            for i, folder in enumerate(matching_folders, 1):
                print(f"\n{'='*70}")
                print(f"[{i}/{len(matching_folders)}] Processing folder: {folder}")
                print(f"{'='*70}\n")
                
                try:
                    folder_stats = await pipeline.process_inbox(
                        imap_config,
                        folder=folder,
                        limit=limit,  # Limit applies per folder
                        new_only=args.new_only,
                        since_date=since_date,
                        full_sync=args.full_sync if hasattr(args, 'full_sync') else False,
                        reset_sync_state=args.reset_sync_state if hasattr(args, 'reset_sync_state') else False
                    )
                    
                    all_stats.append((folder, folder_stats))
                    total_processed += folder_stats.get('processed', 0)
                    print(f"\n✅ {folder}: {folder_stats.get('processed', 0)} emails processed")
                    
                except Exception as e:
                    total_errors += 1
                    print(f"\n❌ Error processing {folder}: {e}")
                    logger.error(f"Error processing folder {folder}: {e}", exc_info=True)
                    # Continue with next folder
                    continue
            
            # Print aggregated summary
            print(f"\n{'='*70}")
            print(f"RECURSIVE PROCESSING SUMMARY")
            print(f"{'='*70}")
            print(f"Base folder: {args.folder}")
            print(f"Folders processed: {len(all_stats)}/{len(matching_folders)}")
            print(f"Total emails processed: {total_processed}")
            if total_errors > 0:
                print(f"Errors: {total_errors}")
            print(f"\nPer-folder breakdown:")
            for folder, stats in all_stats:
                processed = stats.get('processed', 0)
                print(f"  {folder:50} {processed:5d} emails")
            print(f"{'='*70}\n")
            
            stats = {'processed': total_processed, 'folders_processed': len(all_stats)}
            
        else:
            # Single folder processing (existing behavior)
            stats = await pipeline.process_inbox(
                imap_config,
                folder=args.folder,
                limit=limit,
                new_only=args.new_only,
                since_date=since_date,
                full_sync=args.full_sync if hasattr(args, 'full_sync') else False,
                reset_sync_state=args.reset_sync_state if hasattr(args, 'reset_sync_state') else False
            )
        
        print(f"✅ Processing complete!")
        
        # Success message
        if args.dry_run:
            print(f"\n💡 Next step: Run without --dry-run to apply changes")
        else:
            print(f"\n✅ Changes applied to IMAP server")
            print(f"   Check your inbox in Mail.app to see results!")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Interrupted by user")
        pipeline._print_report((datetime.now() - datetime.now()).total_seconds())
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

