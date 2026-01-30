"""
Unified Two-Stage Classifier

Combines application scoring and urgency verification into a single classifier.

Triggers Stage 2 (gpt-5) for:
1. Applications: rec ≥6 AND sci ≥7 (or custom criteria)
2. Urgency: work-urgent OR (work-* categories with urgency ≥8)

Cost Optimization:
- Uses same prompt for both stages (96-99% caching)
- Only runs Stage 2 for critical emails (~10-15% total)
- Combines application + urgency logic efficiently

Usage:
    classifier = UnifiedTwoStageClassifier()
    result = await classifier.classify(email)
    
    if result['stage_2_triggered']:
        print(f"Reason: {result['stage_2_reason']}")
        print(f"Improvements: {result['improvements']}")
"""

from typing import Optional, Dict, Any
from backend.core.email.models import ProcessedEmail
from backend.core.ai.classifier import AIClassifier, AIClassificationResult
from backend.core.ai.two_stage_application_classifier import TwoStageApplicationClassifier
import logging
from pathlib import Path
import yaml
import asyncio
import jsonschema

logger = logging.getLogger(__name__)


# JSON Schema for application config validation
APPLICATION_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "fast_model": {"type": "string"},
        "detailed_model": {"type": "string"},
        "fast_provider": {"type": "string", "enum": ["openai", "azure", "anthropic"]},
        "detailed_provider": {"type": "string", "enum": ["openai", "azure", "anthropic"]},
        "stage_2_criteria": {
            "type": "object",
            "patternProperties": {
                "^application-": {
                    "oneOf": [
                        {"type": "number", "minimum": 1, "maximum": 10},
                        {
                            "type": "object",
                            "properties": {
                                "recommendation_score": {"type": "number", "minimum": 1, "maximum": 10},
                                "scientific_excellence_score": {"type": "number", "minimum": 1, "maximum": 10},
                                "operator": {"type": "string", "enum": ["and", "or", "AND", "OR"]}
                            },
                            "additionalProperties": False
                        }
                    ]
                }
            },
            "additionalProperties": False
        },
        "combine_strategy": {
            "type": "string",
            "enum": ["max", "stage2", "avg"]
        }
    },
    "additionalProperties": False
}

# JSON Schema for urgency config validation
URGENCY_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "fast_model": {"type": "string"},
        "detailed_model": {"type": "string"},
        "fast_provider": {"type": "string", "enum": ["openai", "azure", "anthropic"]},
        "detailed_provider": {"type": "string", "enum": ["openai", "azure", "anthropic"]},
        "stage_2_criteria": {
            "type": "object",
            "patternProperties": {
                ".*": {
                    "type": "object",
                    "properties": {
                        "verify": {"type": "boolean"},
                        "urgency_score": {"type": "number", "minimum": 1, "maximum": 10}
                    },
                    "additionalProperties": False
                }
            },
            "additionalProperties": False
        },
        "combine_strategy": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": ["stage2", "stage1", "max"]},
                "urgency_score": {"type": "string", "enum": ["stage2", "stage1", "max"]},
                "urgency": {"type": "string", "enum": ["stage2", "stage1", "max"]},
                "other_scores": {"type": "string", "enum": ["stage2", "stage1", "max"]}
            },
            "additionalProperties": False
        }
    },
    "additionalProperties": False
}


class UnifiedTwoStageClassifier:
    """
    Unified two-stage classifier for both applications and urgency.
    """
    
    def __init__(
        self,
        fast_model: str = "gpt-5-mini",
        detailed_model: str = "gpt-5",
        fast_provider: Optional[str] = None,
        detailed_provider: Optional[str] = None,
        application_config_path: Optional[str] = None,
        urgency_config_path: Optional[str] = None
    ):
        """
        Initialize unified two-stage classifier.
        
        Args:
            fast_model: Model for Stage 1
            detailed_model: Model for Stage 2
            fast_provider: Provider for Stage 1 ("openai", "azure", or "anthropic")
            detailed_provider: Provider for Stage 2 ("openai", "azure", or "anthropic")
            application_config_path: Path to application_two_stage.yaml
            urgency_config_path: Path to urgency_two_stage.yaml
        """
        # Load configurations first to get provider info if not specified
        self.app_config = self._load_app_config(application_config_path)
        self.urgency_config = self._load_urgency_config(urgency_config_path)
        
        # Set models and providers (config can override defaults)
        self.fast_model = self.app_config.get('fast_model', fast_model)
        self.detailed_model = self.app_config.get('detailed_model', detailed_model)
        # Provider is optional - if None, AIClassifier will auto-detect from llm_endpoints.yaml
        self.fast_provider = fast_provider or self.app_config.get('fast_provider')
        self.detailed_provider = detailed_provider or self.app_config.get('detailed_provider')
        
        # Extract criteria
        self.app_criteria = self.app_config.get('stage_2_criteria', {})
        self.urgency_criteria = self.urgency_config.get('stage_2_criteria', {})
        
        # Statistics
        self.total_classified = 0
        self.stage_2_triggered = 0
        self.stage_2_for_applications = 0
        self.stage_2_for_urgency = 0
        self.stage_2_for_both = 0
        
        # Cost tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        self.total_cached_tokens = 0
        
        # Cache last result for unified interface (thread-safe)
        import threading
        self._thread_local = threading.local()
        
        # Cost tracking
        self._cost_session = None
        self._cost_source = "cli"
        
        logger.info(
            f"UnifiedTwoStageClassifier initialized: "
            f"Stage 1={self.fast_provider}/{self.fast_model}, "
            f"Stage 2={self.detailed_provider}/{self.detailed_model}"
        )
    
    def _load_app_config(self, config_path: Optional[str] = None) -> Dict:
        """Load application two-stage configuration."""
        if config_path:
            config_file = Path(config_path)
        else:
            config_dir = Path(__file__).parent / "config"
            config_file = config_dir / "application_two_stage.yaml"
            local_file = config_dir / "application_two_stage.local.yaml"
            if local_file.exists():
                config_file = local_file
        
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = yaml.safe_load(f)
                
                # Validate config against schema
                try:
                    jsonschema.validate(instance=config, schema=APPLICATION_CONFIG_SCHEMA)
                    logger.info(f"Loaded and validated application config from {config_file}")
                except jsonschema.ValidationError as e:
                    logger.error(
                        f"Application config validation failed at {config_file}: "
                        f"{e.message} (at path: {'.'.join(str(p) for p in e.path)})"
                    )
                    raise ValueError(f"Invalid application config: {e.message}")
                
                return config
            except yaml.YAMLError as e:
                logger.warning(f"Could not parse app config YAML: {e}")
            except Exception as e:
                logger.warning(f"Could not load app config: {e}")
        
        # Default application config
        logger.warning(
            f"Application config not found at {config_file}, using defaults"
        )
        return {
            'stage_2_criteria': TwoStageApplicationClassifier.DEFAULT_STAGE_2_CRITERIA.copy(),
            'combine_strategy': 'max'
        }
    
    def _load_urgency_config(self, config_path: Optional[str] = None) -> Dict:
        """Load urgency two-stage configuration."""
        if config_path:
            config_file = Path(config_path)
        else:
            config_dir = Path(__file__).parent / "config"
            config_file = config_dir / "urgency_two_stage.yaml"
            local_file = config_dir / "urgency_two_stage.local.yaml"
            if local_file.exists():
                config_file = local_file
        
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = yaml.safe_load(f)
                
                # Validate config against schema
                try:
                    jsonschema.validate(instance=config, schema=URGENCY_CONFIG_SCHEMA)
                    logger.info(f"Loaded and validated urgency config from {config_file}")
                except jsonschema.ValidationError as e:
                    logger.error(
                        f"Urgency config validation failed at {config_file}: "
                        f"{e.message} (at path: {'.'.join(str(p) for p in e.path)})"
                    )
                    raise ValueError(f"Invalid urgency config: {e.message}")
                
                return config
            except yaml.YAMLError as e:
                logger.warning(f"Could not parse urgency config YAML: {e}")
            except Exception as e:
                logger.warning(f"Could not load urgency config: {e}")
        
        # Default urgency config (includes invitations!)
        logger.warning(
            f"Urgency config not found at {config_file}, using defaults"
        )
        return {
            'stage_2_criteria': {
                'work-urgent': {'verify': True},
                'work-colleague': {'urgency_score': 8},
                'work-student': {'urgency_score': 8},
                'work-admin': {'urgency_score': 8},
                'work-other': {'urgency_score': 8},
                'invitation-speaking': {'verify': True},
                'invitation-collaboration': {'verify': True}
            },
            'combine_strategy': {
                'category': 'stage2',
                'urgency_score': 'stage2',
                'urgency': 'stage2',
                'other_scores': 'stage2'
            }
        }
    
    def should_run_stage_2(self, result: AIClassificationResult) -> tuple[bool, str]:
        """
        Determine if Stage 2 should run and why.
        
        Returns:
            (should_run, reason)
        """
        reasons = []
        
        # Check application criteria
        app_trigger = self._check_application_criteria(result)
        if app_trigger:
            reasons.append(f"application: {app_trigger}")
        
        # Check urgency criteria
        urgency_trigger = self._check_urgency_criteria(result)
        if urgency_trigger:
            reasons.append(f"urgency: {urgency_trigger}")
        
        if reasons:
            return True, "; ".join(reasons)
        
        return False, ""
    
    def _check_application_criteria(self, result: AIClassificationResult) -> Optional[str]:
        """Check if application criteria met."""
        category = result.category
        
        if category not in self.app_criteria:
            return None
        
        criteria = self.app_criteria[category]
        
        # Simple format
        if isinstance(criteria, (int, float)):
            if result.recommendation_score and result.recommendation_score >= criteria:
                return f"rec={result.recommendation_score}≥{criteria}"
            return None
        
        # Complex format
        if isinstance(criteria, dict):
            rec_thresh = criteria.get('recommendation_score')
            sci_thresh = criteria.get('scientific_excellence_score')
            operator = criteria.get('operator', 'or').lower()
            
            conditions = []
            if rec_thresh and result.recommendation_score:
                if result.recommendation_score >= rec_thresh:
                    conditions.append(f"rec={result.recommendation_score}≥{rec_thresh}")
            
            if sci_thresh and result.scientific_excellence_score:
                if result.scientific_excellence_score >= sci_thresh:
                    conditions.append(f"sci={result.scientific_excellence_score}≥{sci_thresh}")
            
            if not conditions:
                return None
            
            if operator == 'and':
                if len(conditions) == 2:  # Both must be met
                    return f"{conditions[0]} AND {conditions[1]}"
                return None
            else:  # 'or'
                return " OR ".join(conditions) if conditions else None
        
        return None
    
    def _check_urgency_criteria(self, result: AIClassificationResult) -> Optional[str]:
        """Check if urgency criteria met."""
        category = result.category
        
        if category not in self.urgency_criteria:
            return None
        
        criteria = self.urgency_criteria[category]
        
        # Always verify (e.g., work-urgent)
        if isinstance(criteria, dict) and criteria.get('verify'):
            return "verify classification"
        
        # Urgency score threshold
        if isinstance(criteria, dict) and 'urgency_score' in criteria:
            thresh = criteria['urgency_score']
            if result.urgency_score and result.urgency_score >= thresh:
                return f"urgency={result.urgency_score}≥{thresh}"
        
        return None
    
    def _create_fallback_classification(self, email: ProcessedEmail, error_msg: str) -> AIClassificationResult:
        """
        Create a fallback classification when Stage 1 fails.
        
        This prevents total failure when the fast model times out or errors.
        Uses conservative defaults that won't cause data loss or missed urgent emails.
        
        Args:
            email: The email being classified
            error_msg: Error message describing the failure
            
        Returns:
            AIClassificationResult with conservative fallback values
        """
        return AIClassificationResult(
            category="work-other",  # Conservative catch-all category
            confidence=0.0,  # Zero confidence indicates fallback classification
            reasoning=f"Fallback classification due to Stage 1 failure: {error_msg}",
            urgency="normal",  # Conservative middle ground
            urgency_score=5,  # Middle urgency (neither high nor low)
            urgency_reason=f"Default urgency due to classification failure: {error_msg}",
            summary=email.subject or "No subject",
            action_items=[],
            needs_reply=True,  # Conservative: assume reply needed (safer than missing a reply)
            reply_deadline=None,
            reply_suggestion=None,
            sentiment="neutral",
            is_cold_email=False,
            is_followup=False
        )
    
    async def classify(
        self,
        email: ProcessedEmail,
        sender_history: Optional[Dict] = None
    ) -> AIClassificationResult:
        """
        Classify email with optional Stage 2 (unified interface).
        
        This is the SIMPLE interface - returns just the final classification result,
        abstracting away the two-stage implementation details.
        
        Use this when you want a drop-in replacement for AIClassifier.
        For detailed info (Stage 1/2 results, metadata), use get_last_detailed_result()
        after calling this method.
        
        Returns:
            AIClassificationResult - The final classification (same as AIClassifier)
        """
        detailed_result = await self.classify_detailed(email, sender_history)
        # Store in thread-local storage for thread safety
        self._thread_local.last_detailed_result = detailed_result
        return detailed_result['final_result']
    
    def get_last_detailed_result(self) -> Optional[Dict[str, Any]]:
        """
        Get the detailed result from the last classify() call (thread-safe).
        
        Returns None if classify() hasn't been called yet in this thread.
        
        Returns:
            Dict with stage_1_result, stage_2_result, two_stage_metadata, etc.
        """
        return getattr(self._thread_local, 'last_detailed_result', None)
    
    def set_cost_tracker_session(self, db_session, source: str = "cli"):
        """
        Enable cost tracking for both Stage 1 and Stage 2 classifications.
        
        This method passes the database session to internal AIClassifier instances
        so that all API costs are tracked in the database.
        
        Args:
            db_session: SQLAlchemy database session for cost logging
            source: Source of API calls ('cli', 'api', etc.)
        """
        self._cost_session = db_session
        self._cost_source = source
        logger.debug(f"Cost tracking enabled for two-stage classifier (source={source})")
    
    async def classify_detailed(
        self,
        email: ProcessedEmail,
        sender_history: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Classify email with optional Stage 2 (detailed interface).
        
        This is the DETAILED interface - returns full two-stage metadata
        including Stage 1 result, Stage 2 result (if triggered), improvements, etc.
        
        Returns:
            Dict with stage_1_result, stage_2_result (if triggered), final_result,
            stage_2_triggered, stage_2_reason, improvements, two_stage_metadata
        """
        self.total_classified += 1
        
        # Stage 1: Fast classification
        logger.info(f"Stage 1: Classifying with {self.fast_provider}/{self.fast_model}")
        stage_1_classifier = AIClassifier(
            provider=self.fast_provider,
            model=self.fast_model
        )
        
        # Enable cost tracking for Stage 1
        if self._cost_session:
            stage_1_classifier.set_cost_tracker_session(self._cost_session, self._cost_source)
        
        try:
            stage_1_result = await asyncio.wait_for(
                stage_1_classifier.classify(email, sender_history),
                timeout=60.0  # Increased from 30s - some emails are long/complex
            )
            # Accumulate Stage 1 costs
            stage_1_stats = stage_1_classifier.get_usage_stats()
            self.total_prompt_tokens += stage_1_stats.get('total_prompt_tokens', 0)
            self.total_completion_tokens += stage_1_stats.get('total_completion_tokens', 0)
            self.total_cost += stage_1_stats.get('total_cost_usd', 0)
            self.total_cached_tokens += stage_1_stats.get('total_cached_tokens', 0)
        except asyncio.TimeoutError:
            logger.error(f"Stage 1 timeout for email {email.message_id}, using fallback classification")
            # Return fallback classification (similar to AIClassifier.classify fallback)
            stage_1_result = self._create_fallback_classification(email, "Stage 1 timeout after 30s")
        except Exception as e:
            logger.error(f"Stage 1 failed for email {email.message_id}: {e}, using fallback classification")
            # Return fallback classification
            stage_1_result = self._create_fallback_classification(email, f"Stage 1 error: {str(e)}")
        
        # Check if Stage 1 used fallback (confidence = 0.0)
        stage_1_error = None
        if stage_1_result.confidence == 0.0 and "Fallback classification" in stage_1_result.reasoning:
            stage_1_error = stage_1_result.reasoning
        
        # Check if Stage 2 needed
        should_run, reason = self.should_run_stage_2(stage_1_result)
        
        if not should_run:
            logger.debug(f"Stage 2 not needed for {stage_1_result.category}")
            return {
                'stage_1_result': stage_1_result,
                'stage_1_model': self.fast_model,
                'stage_2_result': None,
                'stage_2_model': None,
                'final_result': stage_1_result,
                'stage_2_triggered': False,
                'stage_2_reason': "",
                'improvements': {},
                'two_stage_metadata': {
                    'two_stage_used': False,
                    'stage_2_triggered': False,
                    'stage_1_model': self.fast_model,
                    'stage_1_category': stage_1_result.category,
                    'stage_1_confidence': stage_1_result.confidence,
                    'stage_1_urgency_score': stage_1_result.urgency_score,
                    'stage_1_recommendation_score': stage_1_result.recommendation_score,
                    'stage_1_scientific_excellence_score': stage_1_result.scientific_excellence_score,
                    'stage_2_model': None,
                    'stage_2_reason': None,
                    'stage_2_error': stage_1_error  # Track Stage 1 errors
                }
            }
        
        # Stage 2: Detailed classification
        logger.info(f"Stage 2 TRIGGERED: {reason}")
        self.stage_2_triggered += 1
        
        # Track reason
        if 'application' in reason:
            self.stage_2_for_applications += 1
        if 'urgency' in reason:
            self.stage_2_for_urgency += 1
        if 'application' in reason and 'urgency' in reason:
            self.stage_2_for_both += 1
        
        stage_2_classifier = AIClassifier(
            provider=self.detailed_provider,
            model=self.detailed_model
        )
        
        # Enable cost tracking for Stage 2
        if self._cost_session:
            stage_2_classifier.set_cost_tracker_session(self._cost_session, self._cost_source)
        
        try:
            stage_2_result = await asyncio.wait_for(
                stage_2_classifier.classify(email, sender_history),
                timeout=60.0
            )
            # Accumulate Stage 2 costs
            stage_2_stats = stage_2_classifier.get_usage_stats()
            self.total_prompt_tokens += stage_2_stats.get('total_prompt_tokens', 0)
            self.total_completion_tokens += stage_2_stats.get('total_completion_tokens', 0)
            self.total_cost += stage_2_stats.get('total_cost_usd', 0)
            self.total_cached_tokens += stage_2_stats.get('total_cached_tokens', 0)
        except asyncio.TimeoutError:
            logger.warning(f"Stage 2 timeout for email {email.message_id}, using Stage 1 result")
            # Return Stage 1 result with error metadata (partial failure handling)
            # Combine Stage 1 and Stage 2 errors if both failed
            combined_error = 'timeout'
            if stage_1_error:
                combined_error = f"Stage 1: {stage_1_error}; Stage 2: timeout"
            return {
                'stage_1_result': stage_1_result,
                'stage_1_model': self.fast_model,
                'stage_2_result': None,
                'stage_2_model': self.detailed_model,
                'final_result': stage_1_result,
                'stage_2_triggered': True,
                'stage_2_reason': reason,
                'stage_2_error': 'Stage 2 timeout',
                'improvements': {},
                'two_stage_metadata': {
                    'two_stage_used': True,
                    'stage_2_triggered': True,
                    'stage_1_model': self.fast_model,
                    'stage_1_category': stage_1_result.category,
                    'stage_1_confidence': stage_1_result.confidence,
                    'stage_1_urgency_score': stage_1_result.urgency_score,
                    'stage_1_recommendation_score': stage_1_result.recommendation_score,
                    'stage_1_scientific_excellence_score': stage_1_result.scientific_excellence_score,
                    'stage_2_model': self.detailed_model,
                    'stage_2_reason': reason,
                    'stage_2_error': combined_error
                }
            }
        except Exception as e:
            logger.warning(f"Stage 2 failed for email {email.message_id}: {e}, using Stage 1 result")
            # Return Stage 1 result with error metadata (partial failure handling)
            # Combine Stage 1 and Stage 2 errors if both failed
            combined_error = str(e)
            if stage_1_error:
                combined_error = f"Stage 1: {stage_1_error}; Stage 2: {str(e)}"
            return {
                'stage_1_result': stage_1_result,
                'stage_1_model': self.fast_model,
                'stage_2_result': None,
                'stage_2_model': self.detailed_model,
                'final_result': stage_1_result,
                'stage_2_triggered': True,
                'stage_2_reason': reason,
                'stage_2_error': str(e),
                'improvements': {},
                'two_stage_metadata': {
                    'two_stage_used': True,
                    'stage_2_triggered': True,
                    'stage_1_model': self.fast_model,
                    'stage_1_category': stage_1_result.category,
                    'stage_1_confidence': stage_1_result.confidence,
                    'stage_1_urgency_score': stage_1_result.urgency_score,
                    'stage_1_recommendation_score': stage_1_result.recommendation_score,
                    'stage_1_scientific_excellence_score': stage_1_result.scientific_excellence_score,
                    'stage_2_model': self.detailed_model,
                    'stage_2_reason': reason,
                    'stage_2_error': combined_error
                }
            }
        
        # Combine results
        final_result = self._combine_results(stage_1_result, stage_2_result, reason)
        
        # Calculate improvements
        improvements = self._calculate_improvements(stage_1_result, stage_2_result, final_result)
        
        if improvements:
            logger.info(f"Stage 2 improvements: {improvements}")
        
        # Prepare database metadata
        two_stage_metadata = {
            'two_stage_used': True,
            'stage_2_triggered': True,
            'stage_1_model': self.fast_model,
            'stage_1_category': stage_1_result.category,
            'stage_1_confidence': stage_1_result.confidence,
            'stage_1_urgency_score': stage_1_result.urgency_score,
            'stage_1_recommendation_score': stage_1_result.recommendation_score,
            'stage_1_scientific_excellence_score': stage_1_result.scientific_excellence_score,
            'stage_2_model': self.detailed_model,
            'stage_2_reason': reason,
            'stage_2_error': stage_1_error  # Track Stage 1 errors even when Stage 2 succeeds
        }
        
        return {
            'stage_1_result': stage_1_result,
            'stage_1_model': self.fast_model,
            'stage_2_result': stage_2_result,
            'stage_2_model': self.detailed_model,
            'final_result': final_result,
            'stage_2_triggered': True,
            'stage_2_reason': reason,
            'improvements': improvements,
            'two_stage_metadata': two_stage_metadata
        }
    
    def _combine_results(
        self,
        stage_1: AIClassificationResult,
        stage_2: AIClassificationResult,
        reason: str
    ) -> AIClassificationResult:
        """
        Combine Stage 1 and Stage 2 results intelligently.
        
        Strategy:
        - For applications: Take MAX of scores (per app config)
        - For urgency: Trust Stage 2 (per urgency config)
        - For category: Prefer Stage 2 (more accurate)
        """
        # Start with Stage 2 as base
        combined = stage_2.model_copy(deep=True)
        
        # If triggered for applications, take MAX of application scores
        # BUT only if Stage 2 still classifies as an application
        if 'application' in reason and combined.category.startswith('application-'):
            app_strategy = self.app_config.get('combine_strategy', 'max')
            
            if app_strategy == 'max':
                if stage_1.recommendation_score and stage_2.recommendation_score:
                    combined.recommendation_score = max(
                        stage_1.recommendation_score,
                        stage_2.recommendation_score
                    )
                
                if stage_1.scientific_excellence_score and stage_2.scientific_excellence_score:
                    combined.scientific_excellence_score = max(
                        stage_1.scientific_excellence_score,
                        stage_2.scientific_excellence_score
                    )
                
                if stage_1.relevance_score and stage_2.relevance_score:
                    combined.relevance_score = max(
                        stage_1.relevance_score,
                        stage_2.relevance_score
                    )
            # else: keep Stage 2 values (already in combined)
        elif 'application' in reason:
            # Stage 2 reclassified to non-application - trust Stage 2 completely
            logger.debug(
                f"Stage 1 triggered app logic but Stage 2 reclassified: "
                f"{stage_1.category} → {combined.category}"
            )
        
        # If triggered for urgency, apply urgency combine strategy
        if 'urgency' in reason:
            urgency_strategy = self.urgency_config.get('combine_strategy', {})
            urgency_score_strategy = urgency_strategy.get('urgency_score', 'stage2')
            urgency_level_strategy = urgency_strategy.get('urgency', 'stage2')
            
            # Handle urgency_score
            if urgency_score_strategy == 'max':
                if stage_1.urgency_score and stage_2.urgency_score:
                    combined.urgency_score = max(
                        stage_1.urgency_score,
                        stage_2.urgency_score
                    )
                # Recalculate urgency level based on max score
                if combined.urgency_score:
                    if combined.urgency_score >= 8:
                        combined.urgency = "urgent"
                    elif combined.urgency_score >= 5:
                        combined.urgency = "normal"
                    else:
                        combined.urgency = "low"
            elif urgency_score_strategy == 'stage2':
                # Keep Stage 2 values (already in combined)
                # Don't recalculate - trust Stage 2's assessment
                pass
        
        return combined
    
    def _calculate_improvements(
        self,
        stage_1: AIClassificationResult,
        stage_2: AIClassificationResult,
        final: AIClassificationResult
    ) -> Dict[str, Any]:
        """Track what improved from Stage 1 to final."""
        improvements = {}
        
        # Category change
        if stage_1.category != final.category:
            improvements['category'] = {
                'from': stage_1.category,
                'to': final.category
            }
        
        # Score improvements
        if stage_1.recommendation_score and final.recommendation_score:
            delta = final.recommendation_score - stage_1.recommendation_score
            if delta != 0:
                improvements['recommendation_score'] = delta
        
        if stage_1.scientific_excellence_score and final.scientific_excellence_score:
            delta = final.scientific_excellence_score - stage_1.scientific_excellence_score
            if delta != 0:
                improvements['scientific_excellence_score'] = delta
        
        if stage_1.urgency_score and final.urgency_score:
            delta = final.urgency_score - stage_1.urgency_score
            if delta != 0:
                improvements['urgency_score'] = delta
        
        if stage_1.urgency != final.urgency:
            improvements['urgency'] = {
                'from': stage_1.urgency,
                'to': final.urgency
            }
        
        return improvements
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics on two-stage usage."""
        stage_2_rate = (
            self.stage_2_triggered / self.total_classified
            if self.total_classified > 0 else 0
        )
        
        return {
            'total_classified': self.total_classified,
            'stage_2_triggered': self.stage_2_triggered,
            'stage_2_rate': stage_2_rate,
            'stage_2_for_applications': self.stage_2_for_applications,
            'stage_2_for_urgency': self.stage_2_for_urgency,
            'stage_2_for_both': self.stage_2_for_both
        }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get cumulative usage and cost statistics for the two-stage classifier.
        Mirrors the interface of AIClassifier.get_usage_stats() for compatibility.
        
        Returns:
            Dict with token counts, costs, and stage 2 statistics
        """
        stats = {
            'provider': 'openai',  # Both stages use OpenAI
            'model': f"{self.fast_model} → {self.detailed_model}",
            'total_prompt_tokens': self.total_prompt_tokens,
            'total_completion_tokens': self.total_completion_tokens,
            'total_tokens': self.total_prompt_tokens + self.total_completion_tokens,
            'total_cost_usd': round(self.total_cost, 4),
            'avg_tokens_per_email': round((self.total_prompt_tokens + self.total_completion_tokens) / max(1, self.total_classified), 0) if self.total_classified > 0 else 0,
            # Add cache statistics if available
            'total_cached_tokens': self.total_cached_tokens,
            'cache_hit_rate_percent': round((self.total_cached_tokens / self.total_prompt_tokens) * 100, 1) if self.total_prompt_tokens > 0 else 0,
            'cache_note': 'Cost already discounted by OpenAI (50-90% off cached tokens)',
            # Two-stage specific stats
            'total_classified': self.total_classified,
            'stage_2_triggered': self.stage_2_triggered,
            'stage_2_rate': round((self.stage_2_triggered / self.total_classified) * 100, 1) if self.total_classified > 0 else 0,
            'avg_cost_per_email': round(self.total_cost / max(1, self.total_classified), 6) if self.total_classified > 0 else 0
        }
        return stats


# Example usage
if __name__ == "__main__":
    import asyncio
    from datetime import datetime, timezone
    
    async def test():
        classifier = UnifiedTwoStageClassifier()
        
        # Test urgent work email
        email = ProcessedEmail(
            uid="test_1",
            message_id="<test@test.com>",
            subject="URGENT: Deadline Tomorrow",
            from_address="colleague@university.edu",
            from_name="Dr. Colleague",
            sender_domain="university.edu",
            to_addresses=["you@university.edu"],
            body_text="We need your input by tomorrow for the grant deadline!",
            body_markdown="We need your input by tomorrow for the grant deadline!",
            date=datetime.now(timezone.utc),
            has_attachments=False,
            attachment_count=0
        )
        
        result = await classifier.classify(email)
        
        print(f"Category: {result['final_result'].category}")
        print(f"Urgency: {result['final_result'].urgency} (score={result['final_result'].urgency_score})")
        print(f"Stage 2 triggered: {result['stage_2_triggered']}")
        if result['stage_2_triggered']:
            print(f"Reason: {result['stage_2_reason']}")
            print(f"Improvements: {result['improvements']}")
        
        print(f"\nStats: {classifier.get_stats()}")
    
    asyncio.run(test())

