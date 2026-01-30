"""
VIP Sender Management

Phase 1: Manual VIP list with priority levels
Phase 2: AI-assisted VIP detection from patterns

Manages VIP senders with priority levels for accelerated handling.
"""
import yaml
import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from pathlib import Path

from .models import ProcessedEmail, EmailAction, EmailCategory
from .rule_classifier import ClassificationRule, RuleCondition
from backend.core.paths import get_config_path

logger = logging.getLogger(__name__)


@dataclass
class VIPInfo:
    """Information about a VIP sender"""
    sender: str
    level: str  # urgent, high, medium
    color: int  # Apple Mail color code
    priority: int  # Rule priority (1-3)
    should_forward: bool = False
    forward_address: Optional[str] = None


class VIPManager:
    """
    Manage VIP sender lists and priorities.
    
    Phase 1: Manual VIP list from YAML configuration
    Phase 2: AI-assisted VIP detection and learning
    
    VIP Levels:
    - urgent: Red color (1), priority 1, immediate attention
    - high: Orange color (2), priority 2, same-day response
    - medium: Yellow color (3), priority 3, 2-day response
    """
    
    VIP_LEVELS = {
        'urgent': {'color': 1, 'priority': 1, 'description': 'Immediate attention'},
        'high': {'color': 2, 'priority': 2, 'description': 'Same-day response'},
        'medium': {'color': 3, 'priority': 3, 'description': '2-day response'},
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize VIP manager from configuration file.

        Args:
            config_path: Path to VIP configuration YAML file. If None, uses get_config_path().
        """
        if config_path is None:
            resolved = get_config_path("vip_senders.yaml")
            self.config_path = str(resolved) if resolved else "config/vip_senders.yaml"
        else:
            self.config_path = config_path
        self.vips: Dict[str, Set[str]] = {}  # level -> set of email addresses
        self.vip_domains: Dict[str, Set[str]] = {}  # level -> set of domains
        self.settings: Dict = {}
        self._load_vip_config()
    
    def _load_vip_config(self):
        """Load VIP configuration from YAML file"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.warning(f"VIP config not found: {self.config_path}, using empty VIP list")
                self.vips = {'urgent': set(), 'high': set(), 'medium': set()}
                self.vip_domains = {'urgent': set(), 'high': set(), 'medium': set()}
                self.settings = {}
                return
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Load VIP senders
            vip_senders = config.get('vip_senders', {})
            for level in ['urgent', 'high', 'medium']:
                senders = vip_senders.get(level, [])
                # Normalize to lowercase for case-insensitive matching
                self.vips[level] = {s.lower() for s in senders if s}
            
            # Load VIP domains (optional)
            vip_domains = config.get('vip_domains', {})
            for level in ['urgent', 'high', 'medium']:
                domains = vip_domains.get(level, [])
                self.vip_domains[level] = {d.lower() for d in domains if d}
            
            # Load settings
            self.settings = config.get('vip_settings', {})
            
            total_vips = sum(len(v) for v in self.vips.values())
            logger.info(f"Loaded {total_vips} VIP senders from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load VIP config from {self.config_path}: {e}")
            self.vips = {'urgent': set(), 'high': set(), 'medium': set()}
            self.vip_domains = {'urgent': set(), 'high': set(), 'medium': set()}
            self.settings = {}
    
    def check_vip(self, email: ProcessedEmail) -> Optional[VIPInfo]:
        """
        Check if email is from a VIP sender.
        
        Args:
            email: Processed email to check
            
        Returns:
            VIPInfo if sender is VIP, None otherwise
        """
        sender_lower = email.from_address.lower()
        sender_domain = email.sender_domain.lower()
        
        # Check each VIP level (urgent first)
        for level in ['urgent', 'high', 'medium']:
            # Check exact email match
            if sender_lower in self.vips[level]:
                return self._create_vip_info(email.from_address, level)
            
            # Check domain match
            if sender_domain in self.vip_domains[level]:
                return self._create_vip_info(email.from_address, level)
        
        return None
    
    def _create_vip_info(self, sender: str, level: str) -> VIPInfo:
        """Create VIPInfo object for a VIP sender"""
        config = self.VIP_LEVELS[level]
        
        # Check if should forward
        should_forward = False
        forward_address = None
        if level == 'urgent' and self.settings.get('forward_urgent_vips', False):
            should_forward = True
            forward_address = self.settings.get('forward_address')
        
        return VIPInfo(
            sender=sender,
            level=level,
            color=config['color'],
            priority=config['priority'],
            should_forward=should_forward,
            forward_address=forward_address
        )
    
    def get_vip_rules(self) -> List[ClassificationRule]:
        """
        Generate classification rules for all VIP levels.
        
        Returns:
            List of ClassificationRule objects for VIPs
        """
        rules = []
        
        for level in ['urgent', 'high', 'medium']:
            if not self.vips[level] and not self.vip_domains[level]:
                continue  # Skip empty levels
            
            config = self.VIP_LEVELS[level]
            conditions = []
            
            # Add email address conditions
            if self.vips[level]:
                # Create regex pattern for all VIP senders at this level
                pattern = '|'.join(self.vips[level])
                conditions.append(
                    RuleCondition(
                        field='from',
                        pattern=pattern,
                        match_type='matches',
                        case_sensitive=False
                    )
                )
            
            # Add domain conditions
            if self.vip_domains[level]:
                domain_pattern = '|'.join(self.vip_domains[level])
                conditions.append(
                    RuleCondition(
                        field='domain',
                        pattern=domain_pattern,
                        match_type='matches',
                        case_sensitive=False
                    )
                )
            
            # Create action
            action = EmailAction(
                type='color',
                color=config['color'],
                labels=['VIP', level.upper()]
            )
            
            # Add forwarding if enabled
            if level == 'urgent' and self.settings.get('forward_urgent_vips', False):
                forward_addr = self.settings.get('forward_address')
                if forward_addr:
                    action.forward_to = forward_addr
            
            # Create rule
            rule = ClassificationRule(
                name=f"VIP - {level.upper()} Priority",
                priority=config['priority'],
                category=EmailCategory.WORK,
                action=action,
                conditions=conditions,
                stop_on_match=False  # Allow other rules to also apply
            )
            
            rules.append(rule)
        
        return rules
    
    def is_vip(self, sender: str) -> bool:
        """
        Quick check if sender is any VIP level.
        
        Args:
            sender: Email address to check
            
        Returns:
            True if sender is VIP
        """
        sender_lower = sender.lower()
        for level_senders in self.vips.values():
            if sender_lower in level_senders:
                return True
        return False
    
    def get_vip_level(self, sender: str) -> Optional[str]:
        """
        Get VIP level for sender.
        
        Args:
            sender: Email address to check
            
        Returns:
            VIP level ('urgent', 'high', 'medium') or None
        """
        sender_lower = sender.lower()
        for level, senders in self.vips.items():
            if sender_lower in senders:
                return level
        return None
    
    def get_vip_stats(self) -> Dict[str, int]:
        """
        Get VIP statistics.
        
        Returns:
            Dictionary with counts per level
        """
        return {
            level: len(self.vips[level])
            for level in ['urgent', 'high', 'medium']
        }
    
    def add_vip(self, sender: str, level: str = 'medium'):
        """
        Add a VIP sender (runtime only, not persisted to file).
        
        Args:
            sender: Email address to add
            level: VIP level ('urgent', 'high', 'medium')
        """
        if level not in self.vips:
            logger.warning(f"Invalid VIP level: {level}")
            return
        
        self.vips[level].add(sender.lower())
        logger.info(f"Added VIP: {sender} at level {level}")
    
    def remove_vip(self, sender: str):
        """
        Remove a VIP sender (runtime only, not persisted to file).
        
        Args:
            sender: Email address to remove
        """
        sender_lower = sender.lower()
        for level in self.vips.values():
            if sender_lower in level:
                level.remove(sender_lower)
                logger.info(f"Removed VIP: {sender}")
                return

