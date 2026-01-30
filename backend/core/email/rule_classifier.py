"""
Rule-based email classification system.
Runs before AI classification to handle simple, deterministic cases.

Rules are based on regex patterns matching:
- Sender address/domain
- Recipient address
- Subject line

Rules support AND operations to combine conditions.
"""
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import yaml
import logging

from .models import ProcessedEmail, EmailCategory, EmailAction, AppleMailColor

logger = logging.getLogger(__name__)


@dataclass
class RuleCondition:
    """A single condition in a rule"""
    field: str  # 'from', 'to', 'subject', 'domain'
    pattern: str  # Regex pattern
    match_type: str = 'contains'  # 'contains', 'matches', 'equals'
    case_sensitive: bool = False
    negate: bool = False  # If True, inverts the match result (NOT condition)
    
    def evaluate(self, email: ProcessedEmail) -> bool:
        """
        Evaluate if this condition matches the email.
        
        Args:
            email: Email to evaluate
            
        Returns:
            True if condition matches (or doesn't match if negate=True)
        """
        # Get field value
        if self.field == 'from':
            value = email.from_address
        elif self.field == 'to':
            value = ', '.join(email.to_addresses)
        elif self.field == 'to_cc_combined':
            # Combined To and CC addresses for group email detection
            to_list = email.to_addresses or []
            cc_list = email.cc_addresses or []
            value = ', '.join(to_list + cc_list)
        elif self.field == 'subject':
            value = email.subject
        elif self.field == 'domain':
            value = email.sender_domain
        elif self.field == 'body':
            # Email body content for keyword matching
            value = email.body_text or email.body_markdown or ''
        elif self.field == 'has_attachments':
            # Boolean field - pattern should be 'true' or 'false'
            value = 'true' if email.has_attachments else 'false'
        else:
            logger.warning(f"Unknown field: {self.field}")
            return False
        
        # Apply case sensitivity
        if not self.case_sensitive:
            value = value.lower()
            pattern = self.pattern.lower()
        else:
            pattern = self.pattern
        
        # Evaluate based on match type
        try:
            if self.match_type == 'equals':
                result = value == pattern
            elif self.match_type == 'contains':
                result = pattern in value
            elif self.match_type == 'matches':
                result = bool(re.search(pattern, value))
            else:
                logger.warning(f"Unknown match_type: {self.match_type}")
                return False
            
            # Apply negation if specified
            return not result if self.negate else result
                
        except re.error as e:
            logger.error(f"Invalid regex pattern '{pattern}': {e}")
            return False


@dataclass
class ClassificationRule:
    """
    A classification rule with conditions and actions.
    All conditions must match (AND operation).
    """
    name: str
    conditions: List[RuleCondition]
    category: EmailCategory
    action: EmailAction
    priority: int = 100  # Lower number = higher priority
    stop_on_match: bool = True  # Stop processing rules after match
    application_source: Optional[str] = None  # Source identifier for filtering (e.g., "ai_center")
    
    def matches(self, email: ProcessedEmail) -> bool:
        """
        Check if all conditions match (AND operation).
        
        Args:
            email: Email to evaluate
            
        Returns:
            True if all conditions match
        """
        if not self.conditions:
            return False
        
        # All conditions must match (AND)
        return all(condition.evaluate(email) for condition in self.conditions)


class RuleBasedClassifier:
    """
    Rule-based email classifier.
    Processes rules in priority order before AI classification.
    """
    
    def __init__(self, rules: Optional[List[ClassificationRule]] = None):
        """
        Initialize classifier with rules.
        
        Args:
            rules: List of classification rules (sorted by priority, stable sort)
        """
        # Sort by priority with secondary sort on original order (stable)
        rules_list = rules or []
        self.rules = sorted(enumerate(rules_list), key=lambda x: (x[1].priority, x[0]))
        self.rules = [rule for _, rule in self.rules]
        logger.info(f"Initialized RuleBasedClassifier with {len(self.rules)} rules")
    
    def classify(self, email: ProcessedEmail) -> Optional[tuple[EmailCategory, EmailAction, Optional[str]]]:
        """
        Classify email using rules.
        
        Args:
            email: Email to classify
            
        Returns:
            (category, action, application_source) tuple if rule matches, None otherwise
        """
        first_match = None
        
        for rule in self.rules:
            if rule.matches(email):
                logger.info(f"Email {email.uid} matched rule: {rule.name}")
                logger.debug(f"  Category: {rule.category}, Action: {rule.action.type}")
                
                # Store first match
                if first_match is None:
                    first_match = (rule.category, rule.action, rule.application_source)
                
                # Stop if rule says so
                if rule.stop_on_match:
                    return first_match
        
        return first_match
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'RuleBasedClassifier':
        """
        Load rules from YAML configuration file.
        
        Args:
            yaml_path: Path to YAML file
            
        Returns:
            Configured RuleBasedClassifier
        """
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        rules = []
        for rule_config in config.get('rules', []):
            # Parse conditions
            conditions = []
            for cond in rule_config.get('conditions', []):
                conditions.append(RuleCondition(
                    field=cond['field'],
                    pattern=cond['pattern'],
                    match_type=cond.get('match_type', 'contains'),
                    case_sensitive=cond.get('case_sensitive', False),
                    negate=cond.get('negate', False)
                ))
            
            # Parse action
            action_config = rule_config['action']
            action = EmailAction(
                type=action_config['type'],
                folder=action_config.get('folder'),
                color=action_config.get('color'),
                labels=action_config.get('labels', []),
                forward_to=action_config.get('forward_to'),
                target_account=action_config.get('target_account')  # For cross-account moves
            )
            
            # Create rule
            rule = ClassificationRule(
                name=rule_config['name'],
                conditions=conditions,
                category=EmailCategory(rule_config['category']),
                action=action,
                priority=rule_config.get('priority', 100),
                stop_on_match=rule_config.get('stop_on_match', True),
                application_source=rule_config.get('application_source')
            )
            rules.append(rule)
        
        return cls(rules)
    
    def add_rule(self, rule: ClassificationRule):
        """Add a rule and re-sort by priority"""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority)
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rule by name"""
        original_len = len(self.rules)
        self.rules = [r for r in self.rules if r.name != rule_name]
        return len(self.rules) < original_len
    
    def list_rules(self) -> List[Dict[str, Any]]:
        """Get list of rules as dictionaries"""
        return [
            {
                'name': rule.name,
                'priority': rule.priority,
                'category': rule.category,
                'conditions': len(rule.conditions),
                'action': rule.action.type
            }
            for rule in self.rules
        ]

