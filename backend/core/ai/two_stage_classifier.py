"""
Two-Stage Classification with Auto-Optimization

Implements intelligent email classification with:
1. Fast triage (Stage 1): Quick category determination
2. Detailed scoring (Stage 2): Category-specific model for important emails
3. Auto-optimization (Stage 3): Retry low-confidence classifications with fallback model

This optimizes both cost and accuracy by using expensive models only when needed.
"""

from typing import Optional, Tuple, Dict
from fnmatch import fnmatch
import logging
import asyncio

from .classifier import AIClassifier
from backend.core.email.models import ProcessedEmail
from .config_constants import (
    STAGE_2_CONFIDENCE_THRESHOLD,
    AUTO_OPT_CONFIDENCE_THRESHOLD,
    PERSISTENT_LOW_CONFIDENCE_THRESHOLD,
)

logger = logging.getLogger(__name__)


class TwoStageClassifier:
    """
    Two-stage classification with auto-optimization.
    
    Workflow:
    1. Stage 1 (Fast Triage): Use cheap model to determine category
    2. Stage 2 (Detailed Scoring): Use category-specific model if needed
    3. Stage 3 (Auto-Optimization): Retry with fallback if confidence low
    
    Features:
    - Cost-efficient: Only uses expensive models when necessary
    - High accuracy: Better models for important categories
    - Robust: Auto-retry for low-confidence cases
    - Transparent: Returns metadata about decision process
    """
    
    def __init__(
        self,
        fast_model: str = "gpt-4o-mini",
        fallback_model: str = "claude-3-5-sonnet-20241022",
        confidence_threshold: Optional[float] = None,
        needs_detailed_scoring: Optional[set] = None,
        timeout_seconds: int = 60
    ):
        """
        Initialize two-stage classifier.
        
        Args:
            fast_model: Model for Stage 1 (fast triage)
            fallback_model: Model for Stage 3 (auto-optimization)
            confidence_threshold: Minimum confidence before auto-optimization (default from constants)
            needs_detailed_scoring: Set of category patterns needing Stage 2
            timeout_seconds: Timeout for each classification call
        """
        self.fast_model = fast_model
        self.fallback_model = fallback_model
        self.confidence_threshold = confidence_threshold or AUTO_OPT_CONFIDENCE_THRESHOLD
        self.timeout_seconds = timeout_seconds
        
        # Categories that need detailed scoring (Stage 2)
        # Note: application-* removed - use reprocess_applications.py for manual Stage 2
        self.needs_detailed_scoring = needs_detailed_scoring or {
            "invitation-*",
            "review-*"
        }
        
        logger.info(
            f"TwoStageClassifier initialized: "
            f"fast={fast_model}, fallback={fallback_model}, "
            f"threshold={confidence_threshold}"
        )
    
    def _should_use_detailed_scoring(self, category: str, confidence: float) -> bool:
        """
        Determine if email needs detailed scoring (Stage 2).
        
        Criteria:
        1. Category matches detailed scoring patterns
        2. OR confidence is low (could be misclassified)
        
        Args:
            category: Email category from Stage 1
            confidence: Confidence score from Stage 1
            
        Returns:
            True if detailed scoring needed
        """
        # Check category patterns
        for pattern in self.needs_detailed_scoring:
            if fnmatch(category, pattern):
                logger.debug(f"Category {category} matches pattern {pattern}, needs Stage 2")
                return True
        
        # Low confidence suggests uncertainty - use better model
        if confidence < STAGE_2_CONFIDENCE_THRESHOLD:
            logger.debug(
                f"Low confidence ({confidence:.2f}) for {category}, "
                f"needs Stage 2 for verification (threshold: {STAGE_2_CONFIDENCE_THRESHOLD})"
            )
            return True
        
        return False
    
    async def classify(
        self,
        email: ProcessedEmail,
        sender_history: Optional[dict] = None
    ) -> Tuple[any, Dict]:
        """
        Classify email with two-stage approach and auto-optimization.
        
        Args:
            email: Email to classify
            sender_history: Optional sender history context
            
        Returns:
            Tuple of (final_result, metadata)
            
            metadata contains:
            - two_stage: bool - Always True
            - initial_category: str - Category from Stage 1
            - initial_confidence: float - Confidence from Stage 1
            - models_used: List[str] - All models used
            - stage_2_skipped: bool - Whether Stage 2 was skipped
            - final_category: str - Final category (may differ from initial)
            - final_confidence: float - Final confidence score
            - category_override: Dict - Details if category changed
            - auto_optimized: bool - Whether Stage 3 was triggered
            - persistent_low_confidence: bool - If both models had low confidence
            - fallback_confidence: float - Confidence from fallback model (if used)
            - final_model: str - Model that produced final result
        """
        metadata = {
            "two_stage": True,
            "models_used": [],
            "auto_optimized": False,
            "persistent_low_confidence": False
        }
        
        # ============================================================
        # STAGE 1: Fast Classification
        # ============================================================
        logger.info(f"Stage 1: Fast classification with {self.fast_model}")
        fast_classifier = AIClassifier(
            provider="openai",
            model=self.fast_model,
            temperature=0.1
        )
        
        try:
            initial_result = await asyncio.wait_for(
                fast_classifier.classify(email, sender_history),
                timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.error(f"Stage 1 timeout after {self.timeout_seconds}s")
            raise TimeoutError(f"Stage 1 classification timed out after {self.timeout_seconds}s")
        
        metadata["initial_category"] = initial_result.category
        metadata["initial_confidence"] = initial_result.confidence
        metadata["models_used"].append(self.fast_model)
        
        logger.info(
            f"Stage 1 result: {initial_result.category} "
            f"(confidence: {initial_result.confidence:.2f})"
        )
        
        # Check if we need detailed scoring
        if not self._should_use_detailed_scoring(
            initial_result.category,
            initial_result.confidence
        ):
            logger.info(
                f"Skipping Stage 2 for {initial_result.category} "
                f"(confidence: {initial_result.confidence:.2f})"
            )
            metadata["stage_2_skipped"] = True
            metadata["final_category"] = initial_result.category
            metadata["final_confidence"] = initial_result.confidence
            metadata["final_model"] = self.fast_model
            return initial_result, metadata
        
        # ============================================================
        # STAGE 2: Detailed Classification with Category-Specific Model
        # ============================================================
        logger.info(
            f"Stage 2: Detailed scoring for {initial_result.category}"
        )
        
        detailed_classifier = AIClassifier(category_hint=initial_result.category)
        try:
            final_result = await asyncio.wait_for(
                detailed_classifier.classify(email, sender_history),
                timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.error(f"Stage 2 timeout after {self.timeout_seconds}s, using Stage 1 result")
            # Fallback to Stage 1 result
            metadata["stage_2_timeout"] = True
            metadata["final_category"] = initial_result.category
            metadata["final_confidence"] = initial_result.confidence
            metadata["final_model"] = self.fast_model
            return initial_result, metadata
        
        metadata["final_category"] = final_result.category
        metadata["final_confidence"] = final_result.confidence
        metadata["models_used"].append(detailed_classifier.model_name)
        
        logger.info(
            f"Stage 2 result: {final_result.category} "
            f"(confidence: {final_result.confidence:.2f}, "
            f"model: {detailed_classifier.model_name})"
        )
        
        # Check if category changed
        if initial_result.category != final_result.category:
            metadata["category_override"] = {
                "from": initial_result.category,
                "to": final_result.category,
                "confidence_delta": final_result.confidence - initial_result.confidence
            }
            logger.warning(
                f"Category override: {initial_result.category} → {final_result.category} "
                f"(Δconf: {final_result.confidence - initial_result.confidence:+.2f})"
            )
        
        # ============================================================
        # STAGE 3: Auto-Optimization (if confidence low)
        # ============================================================
        if final_result.confidence < self.confidence_threshold:
            logger.warning(
                f"Low confidence ({final_result.confidence:.2f}), "
                f"triggering Stage 3 with fallback model: {self.fallback_model}"
            )
            
            fallback_classifier = AIClassifier(
                model=self.fallback_model,
                category_hint=final_result.category
            )
            fallback_result = await fallback_classifier.classify(email, sender_history)
            
            metadata["models_used"].append(self.fallback_model)
            metadata["fallback_confidence"] = fallback_result.confidence
            
            logger.info(
                f"Stage 3 result: {fallback_result.category} "
                f"(confidence: {fallback_result.confidence:.2f})"
            )
            
            # Check for persistent low confidence (both models uncertain)
            if (final_result.confidence < PERSISTENT_LOW_CONFIDENCE_THRESHOLD and
                fallback_result.confidence < PERSISTENT_LOW_CONFIDENCE_THRESHOLD):
                metadata["persistent_low_confidence"] = True
                logger.warning(
                    f"⚠️  Persistent low confidence detected for email {email.uid}:\n"
                    f"   Primary model ({detailed_classifier.model_name}): {final_result.confidence:.2f}\n"
                    f"   Fallback model ({self.fallback_model}): {fallback_result.confidence:.2f}\n"
                    f"   Threshold: {PERSISTENT_LOW_CONFIDENCE_THRESHOLD}\n"
                    f"   This email may need manual review."
                )
            
            # Use result with higher confidence
            if fallback_result.confidence > final_result.confidence:
                confidence_improvement = fallback_result.confidence - final_result.confidence
                logger.info(
                    f"✓ Fallback model improved confidence: "
                    f"{final_result.confidence:.2f} → {fallback_result.confidence:.2f} "
                    f"(+{confidence_improvement:.2f})"
                )
                metadata["auto_optimized"] = True
                final_result = fallback_result
                metadata["final_category"] = fallback_result.category
                metadata["final_confidence"] = fallback_result.confidence
            else:
                logger.info(
                    f"Keeping original result (confidence not improved by fallback)"
                )
        
        # Set final model
        metadata["final_model"] = (
            self.fallback_model if metadata["auto_optimized"]
            else detailed_classifier.model_name
        )
        
        logger.info(
            f"Classification complete: {metadata['final_category']} "
            f"(confidence: {metadata['final_confidence']:.2f}, "
            f"models: {', '.join(metadata['models_used'])})"
        )
        
        return final_result, metadata
    
    def get_stats(self) -> dict:
        """Get two-stage classifier configuration."""
        return {
            "fast_model": self.fast_model,
            "fallback_model": self.fallback_model,
            "confidence_threshold": self.confidence_threshold,
            "needs_detailed_scoring": list(self.needs_detailed_scoring)
        }

