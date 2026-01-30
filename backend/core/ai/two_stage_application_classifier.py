"""
Two-Stage Application Classifier

Optimized for critical application emails:
- Stage 1: gpt-5-mini (fast, conservative scoring)
- Stage 2: gpt-5 (high-quality scoring for promising candidates)

Strategy:
- Use SAME prompt for both stages (consistency)
- Take MAX of scores (optimistic bias to avoid missing talent)
- Only run Stage 2 for high-potential applicants (cost optimization)

Triggers for Stage 2:
- application-phd with recommendation_score >= 5
- application-postdoc with recommendation_score >= 5
- application-bsc-msc-thesis with recommendation_score >= 6
"""

from typing import Optional, Dict, Any
from backend.core.email.models import ProcessedEmail
from backend.core.ai.classifier import AIClassifier, AIClassificationResult
import logging

logger = logging.getLogger(__name__)


class TwoStageApplicationClassifier:
    """
    Two-stage classifier optimized for identifying top application candidates.
    """
    
    # Default criteria for triggering Stage 2 (high-quality scoring)
    DEFAULT_STAGE_2_CRITERIA = {
        'application-phd': 5,           # Score >= 5
        'application-postdoc': 5,       # Score >= 5
        'application-bsc-msc-thesis': 6, # Score >= 6
        'application-visiting': 5,      # Score >= 5 (optional)
        'application-intern': 6,        # Score >= 6 (optional - typically lower priority)
    }
    
    def __init__(
        self,
        fast_model: str = "gpt-5-mini",
        detailed_model: str = "gpt-5",
        config_path: Optional[str] = None,
        stage_2_criteria: Optional[Dict[str, int]] = None
    ):
        """
        Initialize two-stage application classifier.
        
        Args:
            fast_model: Model for Stage 1 (fast classification)
            detailed_model: Model for Stage 2 (detailed scoring)
            config_path: Optional path to model_routing.yaml
            stage_2_criteria: Optional dict of {category: min_score} for Stage 2 trigger
                             If None, uses DEFAULT_STAGE_2_CRITERIA
                             
        Examples:
            # Use defaults
            classifier = TwoStageApplicationClassifier()
            
            # Custom thresholds (more selective)
            classifier = TwoStageApplicationClassifier(
                stage_2_criteria={
                    'application-phd': 6,      # Higher threshold
                    'application-postdoc': 6,
                    # Don't run Stage 2 for interns/thesis
                }
            )
            
            # Run Stage 2 for ALL applications
            classifier = TwoStageApplicationClassifier(
                stage_2_criteria={
                    'application-phd': 1,
                    'application-postdoc': 1,
                    'application-intern': 1,
                    'application-bsc-msc-thesis': 1,
                    'application-visiting': 1,
                }
            )
        """
        self.fast_model = fast_model
        self.detailed_model = detailed_model
        self.config_path = config_path
        
        # Use custom criteria or load from config
        if stage_2_criteria:
            self.stage_2_criteria = stage_2_criteria
        else:
            # Try to load from config file
            self.stage_2_criteria = self._load_criteria_from_config(config_path)
        
        # Track statistics
        self.total_applications = 0
        self.stage_2_triggered = 0
        self.scores_improved = 0
        self.scores_unchanged = 0
        
        logger.info(
            f"TwoStageApplicationClassifier initialized: "
            f"Stage 1={fast_model}, Stage 2={detailed_model}, "
            f"Criteria={list(self.stage_2_criteria.keys())}"
        )
    
    def _load_criteria_from_config(self, config_path: Optional[str] = None) -> Dict[str, int]:
        """
        Load Stage 2 criteria from YAML config file.
        
        Args:
            config_path: Optional path to application_two_stage.yaml
            
        Returns:
            Dict of {category: min_score} criteria
        """
        from pathlib import Path
        import yaml
        
        # Determine config path
        if config_path:
            config_file = Path(config_path)
        else:
            # Default to application_two_stage.yaml in config directory
            config_dir = Path(__file__).parent / "config"
            config_file = config_dir / "application_two_stage.yaml"
            
            # Check for .local override
            local_config = config_dir / "application_two_stage.local.yaml"
            if local_config.exists():
                config_file = local_config
        
        # Load config if exists
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = yaml.safe_load(f)
                
                if config and 'stage_2_criteria' in config:
                    logger.info(f"Loaded Stage 2 criteria from {config_file}")
                    
                    # Also load models if specified
                    if 'fast_model' in config and not self.fast_model:
                        self.fast_model = config['fast_model']
                    if 'detailed_model' in config and not self.detailed_model:
                        self.detailed_model = config['detailed_model']
                    
                    return config['stage_2_criteria']
            except Exception as e:
                logger.warning(f"Could not load config from {config_file}: {e}")
        
        # Fallback to defaults
        logger.debug("Using default Stage 2 criteria")
        return self.DEFAULT_STAGE_2_CRITERIA.copy()
    
    def should_run_stage_2(self, result: AIClassificationResult) -> bool:
        """
        Determine if Stage 2 (detailed scoring) should be run.
        
        Supports both simple threshold and complex score expressions:
        - Simple: stage_2_criteria = {'application-phd': 5}
          → recommendation_score >= 5
        
        - Complex: stage_2_criteria = {
              'application-phd': {
                  'recommendation_score': 5,
                  'scientific_excellence_score': 5,
                  'operator': 'or'
              }
          }
          → recommendation_score >= 5 OR scientific_excellence_score >= 5
        
        Args:
            result: Stage 1 classification result
            
        Returns:
            True if Stage 2 should run, False otherwise
        """
        category = result.category
        
        # Check if it's an application category we care about
        if category not in self.stage_2_criteria:
            logger.debug(f"{category} not in Stage 2 criteria, skipping")
            return False
        
        criteria = self.stage_2_criteria[category]
        
        # Simple format: just a number (threshold for recommendation_score)
        if isinstance(criteria, (int, float)):
            rec_score = result.recommendation_score
            if rec_score is None:
                logger.warning(f"No recommendation_score for {category}, skipping Stage 2")
                return False
            
            should_run = rec_score >= criteria
            
            if should_run:
                logger.info(f"Stage 2 triggered: {category} rec_score={rec_score} >= {criteria}")
            else:
                logger.debug(f"Stage 2 skipped: {category} rec_score={rec_score} < {criteria}")
            
            return should_run
        
        # Complex format: dict with multiple score conditions
        if isinstance(criteria, dict):
            rec_threshold = criteria.get('recommendation_score')
            sci_threshold = criteria.get('scientific_excellence_score')
            operator = criteria.get('operator', 'or').lower()
            
            conditions_met = []
            score_info = []
            
            # Check recommendation_score
            if rec_threshold is not None:
                rec_score = result.recommendation_score
                if rec_score is not None:
                    meets = rec_score >= rec_threshold
                    conditions_met.append(meets)
                    score_info.append(f"rec={rec_score}>={rec_threshold}:{meets}")
            
            # Check scientific_excellence_score
            if sci_threshold is not None:
                sci_score = result.scientific_excellence_score
                if sci_score is not None:
                    meets = sci_score >= sci_threshold
                    conditions_met.append(meets)
                    score_info.append(f"sci={sci_score}>={sci_threshold}:{meets}")
            
            # Apply operator
            if not conditions_met:
                logger.warning(f"No valid scores for {category}, skipping Stage 2")
                return False
            
            if operator == 'and':
                should_run = all(conditions_met)
            else:  # 'or' (default)
                should_run = any(conditions_met)
            
            if should_run:
                logger.info(
                    f"Stage 2 triggered: {category} ({' {operator.upper()} '.join(score_info)})"
                )
            else:
                logger.debug(
                    f"Stage 2 skipped: {category} ({' {operator.upper()} '.join(score_info)})"
                )
            
            return should_run
        
        # Unknown format
        logger.error(f"Invalid criteria format for {category}: {criteria}")
        return False
    
    async def classify(
        self,
        email: ProcessedEmail,
        sender_history: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Two-stage classification for application emails.
        
        Args:
            email: Email to classify
            sender_history: Optional sender history
            
        Returns:
            Dict with:
                - stage_1_result: Initial classification
                - stage_2_result: Detailed scoring (if triggered)
                - final_result: Combined result with max scores
                - stage_2_triggered: Boolean
                - improvements: Dict of score improvements
        """
        # Stage 1: Fast classification
        logger.info(f"Stage 1: Classifying with {self.fast_model}")
        stage_1_classifier = AIClassifier(
            model=self.fast_model,
            config_path=self.config_path
        )
        
        stage_1_result = await stage_1_classifier.classify(email, sender_history)
        
        # Check if it's an application
        if not stage_1_result.category.startswith('application-'):
            logger.debug(f"Not an application ({stage_1_result.category}), skipping Stage 2")
            return {
                'stage_1_result': stage_1_result,
                'stage_2_result': None,
                'final_result': stage_1_result,
                'stage_2_triggered': False,
                'improvements': {}
            }
        
        self.total_applications += 1
        
        # Check if Stage 2 should run
        if not self.should_run_stage_2(stage_1_result):
            logger.info(
                f"Application detected ({stage_1_result.category}, "
                f"score={stage_1_result.recommendation_score}) but below threshold, "
                f"skipping Stage 2"
            )
            return {
                'stage_1_result': stage_1_result,
                'stage_2_result': None,
                'final_result': stage_1_result,
                'stage_2_triggered': False,
                'improvements': {}
            }
        
        # Stage 2: Detailed scoring with high-quality model
        logger.info(
            f"Stage 2 TRIGGERED: {stage_1_result.category} "
            f"(rec_score={stage_1_result.recommendation_score})"
        )
        self.stage_2_triggered += 1
        
        stage_2_classifier = AIClassifier(
            model=self.detailed_model,
            config_path=self.config_path
        )
        
        stage_2_result = await stage_2_classifier.classify(email, sender_history)
        
        # Combine results: Take MAX of key scores
        final_result = self._combine_results(stage_1_result, stage_2_result)
        
        # Track improvements
        improvements = self._calculate_improvements(stage_1_result, stage_2_result, final_result)
        
        if improvements['any_improvement']:
            self.scores_improved += 1
            logger.info(f"Scores improved: {improvements}")
        else:
            self.scores_unchanged += 1
        
        return {
            'stage_1_result': stage_1_result,
            'stage_2_result': stage_2_result,
            'final_result': final_result,
            'stage_2_triggered': True,
            'improvements': improvements
        }
    
    def _combine_results(
        self,
        stage_1: AIClassificationResult,
        stage_2: AIClassificationResult
    ) -> AIClassificationResult:
        """
        Combine Stage 1 and Stage 2 results, taking MAX of scores.
        
        Strategy:
        - Category: Prefer Stage 2 (more accurate model)
        - Scores: Take MAX (optimistic bias)
        - Other fields: Prefer Stage 2 (more detailed analysis)
        
        Args:
            stage_1: Fast model result
            stage_2: Detailed model result
            
        Returns:
            Combined result
        """
        # Start with Stage 2 as base (more accurate)
        combined = stage_2.model_copy(deep=True)
        
        # Take MAX of critical scores
        if stage_1.recommendation_score and stage_2.recommendation_score:
            combined.recommendation_score = max(
                stage_1.recommendation_score,
                stage_2.recommendation_score
            )
        elif stage_1.recommendation_score:
            combined.recommendation_score = stage_1.recommendation_score
        
        if stage_1.scientific_excellence_score and stage_2.scientific_excellence_score:
            combined.scientific_excellence_score = max(
                stage_1.scientific_excellence_score,
                stage_2.scientific_excellence_score
            )
        elif stage_1.scientific_excellence_score:
            combined.scientific_excellence_score = stage_1.scientific_excellence_score
        
        # For relevance, also take max
        if stage_1.relevance_score and stage_2.relevance_score:
            combined.relevance_score = max(
                stage_1.relevance_score,
                stage_2.relevance_score
            )
        
        # Confidence: Take Stage 2 (more reliable model)
        # Category: Take Stage 2 (more accurate)
        # Reasoning: Take Stage 2 only (don't pollute with Stage 1)
        # Note: We keep Stage 2 reasoning as-is to maintain clean output
        
        return combined
    
    def _calculate_improvements(
        self,
        stage_1: AIClassificationResult,
        stage_2: AIClassificationResult,
        final: AIClassificationResult
    ) -> Dict[str, Any]:
        """Calculate what improved from Stage 1 to final result."""
        improvements = {
            'recommendation_delta': 0,
            'scientific_excellence_delta': 0,
            'relevance_delta': 0,
            'category_changed': stage_1.category != stage_2.category,
            'any_improvement': False
        }
        
        # Calculate deltas
        if stage_1.recommendation_score and final.recommendation_score:
            improvements['recommendation_delta'] = (
                final.recommendation_score - stage_1.recommendation_score
            )
        
        if stage_1.scientific_excellence_score and final.scientific_excellence_score:
            improvements['scientific_excellence_delta'] = (
                final.scientific_excellence_score - stage_1.scientific_excellence_score
            )
        
        if stage_1.relevance_score and final.relevance_score:
            improvements['relevance_delta'] = (
                final.relevance_score - stage_1.relevance_score
            )
        
        # Any improvement?
        improvements['any_improvement'] = (
            improvements['recommendation_delta'] > 0 or
            improvements['scientific_excellence_delta'] > 0 or
            improvements['relevance_delta'] > 0 or
            improvements['category_changed']
        )
        
        return improvements
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics on two-stage performance."""
        stage_2_rate = (
            self.stage_2_triggered / self.total_applications
            if self.total_applications > 0 else 0
        )
        
        improvement_rate = (
            self.scores_improved / self.stage_2_triggered
            if self.stage_2_triggered > 0 else 0
        )
        
        return {
            'total_applications': self.total_applications,
            'stage_2_triggered': self.stage_2_triggered,
            'stage_2_rate': stage_2_rate,
            'scores_improved': self.scores_improved,
            'scores_unchanged': self.scores_unchanged,
            'improvement_rate': improvement_rate
        }


# Example usage
async def example_usage():
    """Example of how to use TwoStageApplicationClassifier."""
    from backend.core.email.models import ProcessedEmail
    from datetime import datetime, timezone
    
    # Initialize
    classifier = TwoStageApplicationClassifier(
        fast_model="gpt-5-mini",
        detailed_model="gpt-5"
    )
    
    # Create test email
    email = ProcessedEmail(
        uid="test_123",
        message_id="<test@example.com>",
        subject="PhD Application in Machine Learning",
        from_address="student@university.edu",
        from_name="Jane Student",
        sender_domain="university.edu",
        to_addresses=["professor@university.edu"],
        body_text="Dear Professor, I am writing to apply for a PhD position...",
        body_markdown="Dear Professor, I am writing to apply for a PhD position...",
        date=datetime.now(timezone.utc),
        has_attachments=False,
        attachment_count=0
    )
    
    # Classify
    result = await classifier.classify(email)
    
    print(f"Stage 1: {result['stage_1_result'].category} "
          f"(rec={result['stage_1_result'].recommendation_score})")
    
    if result['stage_2_triggered']:
        print(f"Stage 2: {result['stage_2_result'].category} "
              f"(rec={result['stage_2_result'].recommendation_score})")
        print(f"Final: {result['final_result'].category} "
              f"(rec={result['final_result'].recommendation_score})")
        print(f"Improvements: {result['improvements']}")
    
    # Get stats
    stats = classifier.get_stats()
    print(f"\nStats: {stats}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())

