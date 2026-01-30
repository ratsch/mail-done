"""
Email preprocessing and normalization system.
Handles special cases like auto-forwarded emails, list emails, etc.

Preprocessing rules transform emails BEFORE classification:
- Extract original sender from forwarded emails
- Unwrap mailing list emails
- Normalize subjects (remove Re:, Fwd: prefixes if needed)
- Extract original recipients
"""
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import yaml
import logging

from .models import ProcessedEmail

logger = logging.getLogger(__name__)


def is_valid_email(email_str: str) -> bool:
    """
    Validate email address format.
    More permissive - handles real-world addresses including escaped chars.
    """
    if not email_str or '@' not in email_str:
        return False
    
    # Clean up common issues from forwarded email bodies
    email_str = email_str.strip('"').strip("'").strip()
    
    # Extract email from angle brackets if present: "Name <email@domain.com>" or "<email@domain.com>"
    if '<' in email_str and '>' in email_str:
        start = email_str.find('<')
        end = email_str.find('>')
        if start < end:
            email_str = email_str[start+1:end]
    
    # Remove escape backslashes (\_  → _)
    email_str = email_str.replace('\\_', '_')
    
    # Handle "at" substitutions sometimes used in forwarded emails
    # e.g., "user_at_domain.com" → "user@domain.com"
    # But only if there's no @ already
    if '@' not in email_str and '_at_' in email_str:
        email_str = email_str.replace('_at_', '@', 1)
    
    # More permissive pattern: allows +, _, -, ., % in local part
    # Allows subdomains and international TLDs
    pattern = r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$'
    
    return bool(re.match(pattern, email_str))


@dataclass
class PreprocessingRule:
    """
    A preprocessing rule that modifies email headers before classification.
    """
    name: str
    description: str
    priority: int = 100
    
    # Conditions to match (when to apply this preprocessing)
    match_from: Optional[str] = None  # Regex pattern
    match_to: Optional[str] = None  # Regex pattern
    match_subject: Optional[str] = None  # Regex pattern
    
    # Transformations to apply
    extract_original_from: Optional[str] = None  # Header name to read from
    extract_original_to: Optional[str] = None  # Header name to read from
    extract_original_subject: Optional[str] = None  # Header name to read from
    
    # Body parsing (for Apple Mail / Gmail style forwarding)
    parse_forwarded_body: bool = False  # Extract original from/to/subject from body
    
    # Header mappings (custom)
    header_mappings: Optional[Dict[str, str]] = None
    
    # Subject transformations
    remove_prefix: Optional[str] = None  # Regex pattern to remove from subject
    
    def matches(self, email: ProcessedEmail, raw_headers: Dict[str, str]) -> bool:
        """
        Check if this preprocessing rule should be applied.
        
        Args:
            email: Processed email
            raw_headers: Raw email headers (for checking original headers)
            
        Returns:
            True if rule conditions match
        """
        # Check from pattern
        if self.match_from:
            if not re.search(self.match_from, email.from_address, re.IGNORECASE):
                return False
        
        # Check to pattern
        if self.match_to:
            to_string = ', '.join(email.to_addresses)
            if not re.search(self.match_to, to_string, re.IGNORECASE):
                return False
        
        # Check subject pattern
        if self.match_subject:
            if not re.search(self.match_subject, email.subject, re.IGNORECASE):
                return False
        
        return True
    
    def apply(self, email: ProcessedEmail, raw_headers: Dict[str, str]) -> ProcessedEmail:
        """
        Apply preprocessing transformations to email.
        
        Args:
            email: Email to transform
            raw_headers: Raw email headers
            
        Returns:
            Transformed email
        """
        # Parse forwarded body (Apple Mail / Gmail style)
        if self.parse_forwarded_body:
            extracted = self._parse_forwarded_email_body(email.body_markdown)
            if extracted:
                # Store original before replacing
                if not email.was_preprocessed:
                    email.original_from = email.from_address
                    email.was_preprocessed = True
                
                if extracted.get('from'):
                    new_from = extracted['from']
                    # Validate before replacing
                    if is_valid_email(new_from):
                        logger.info(f"Preprocessing (body parse): Replacing from '{email.from_address}' with '{new_from}'")
                        email.from_address = new_from
                        email.sender_domain = new_from.split('@')[-1]
                    else:
                        logger.warning(f"Preprocessing: Invalid email extracted from body: '{new_from}', keeping original")
                
                if extracted.get('subject'):
                    logger.info(f"Preprocessing (body parse): Replacing subject with '{extracted['subject']}'")
                    email.subject = extracted['subject']
                
                if extracted.get('to'):
                    logger.info(f"Preprocessing (body parse): Replacing to with '{extracted['to']}'")
                    email.to_addresses = [extracted['to']]
        
        # Extract original sender if specified (from headers)
        if self.extract_original_from:
            original_from = raw_headers.get(self.extract_original_from)
            if original_from:
                # Clean and extract email from angle brackets if present
                cleaned_email = original_from.strip('"').strip("'").strip()
                if '<' in cleaned_email and '>' in cleaned_email:
                    start = cleaned_email.find('<')
                    end = cleaned_email.find('>')
                    if start < end:
                        cleaned_email = cleaned_email[start+1:end]
                
                # Validate before replacing
                if is_valid_email(original_from):
                    if not email.was_preprocessed:
                        email.original_from = email.from_address
                        email.was_preprocessed = True
                    
                    logger.info(f"Preprocessing: Replacing from '{email.from_address}' with original '{cleaned_email}'")
                    email.from_address = cleaned_email
                    # Update domain too
                    email.sender_domain = cleaned_email.split('@')[-1]
                else:
                    logger.warning(f"Preprocessing: Invalid email in header {self.extract_original_from}: '{original_from}', keeping original")
        
        # Extract original recipient
        if self.extract_original_to:
            original_to = raw_headers.get(self.extract_original_to)
            if original_to:
                logger.info(f"Preprocessing: Replacing to with original '{original_to}'")
                email.to_addresses = [original_to]
        
        # Extract original subject
        if self.extract_original_subject:
            original_subject = raw_headers.get(self.extract_original_subject)
            if original_subject:
                logger.info(f"Preprocessing: Replacing subject with original '{original_subject}'")
                email.subject = original_subject
        
        # Apply custom header mappings
        if self.header_mappings:
            for target_field, header_name in self.header_mappings.items():
                value = raw_headers.get(header_name)
                if value:
                    if target_field == 'from':
                        email.from_address = value
                        email.sender_domain = value.split('@')[-1] if '@' in value else email.sender_domain
                    elif target_field == 'subject':
                        email.subject = value
                    elif target_field == 'to':
                        email.to_addresses = [value]
        
        # Remove subject prefix if specified
        if self.remove_prefix:
            email.subject = re.sub(self.remove_prefix, '', email.subject, flags=re.IGNORECASE).strip()
        
        return email
    
    def _parse_forwarded_email_body(self, body: str) -> Optional[Dict[str, str]]:
        """
        Parse forwarded email body to extract original sender, subject, etc.
        
        Handles formats from:
        - Apple Mail: "Begin forwarded message:"
        - Gmail: "---------- Forwarded message ---------"
        - Outlook: "From: ... Sent: ... To: ..."
        
        Args:
            body: Email body text (markdown or plain text)
            
        Returns:
            Dictionary with 'from', 'to', 'subject' if found
        """
        extracted = {}
        
        # Pattern 1: Apple Mail format
        # Begin forwarded message:
        # From: Name <email@domain.com>
        # Subject: ...
        # Date: ...
        # To: ...
        apple_mail_pattern = r'Begin forwarded message:.*?From:\s*([^\n]+).*?Subject:\s*([^\n]+)'
        match = re.search(apple_mail_pattern, body, re.DOTALL | re.IGNORECASE)
        if match:
            from_line = match.group(1).strip()
            subject_line = match.group(2).strip()
            
            # Extract email from "Name <email>" or just "email"
            email_match = re.search(r'<([^>]+)>|([^\s<>]+@[^\s<>]+)', from_line)
            if email_match:
                extracted['from'] = email_match.group(1) or email_match.group(2)
            
            extracted['subject'] = subject_line
            
            # Try to extract To:
            to_match = re.search(r'To:\s*([^\n]+)', body)
            if to_match:
                to_line = to_match.group(1).strip()
                to_email_match = re.search(r'<([^>]+)>|([^\s<>]+@[^\s<>]+)', to_line)
                if to_email_match:
                    extracted['to'] = to_email_match.group(1) or to_email_match.group(2)
            
            return extracted
        
        # Pattern 2: Gmail format
        # ---------- Forwarded message ---------
        # From: Name <email@domain.com>
        # Date: ...
        # Subject: ...
        # To: ...
        gmail_pattern = r'[-]+\s*Forwarded message\s*[-]+.*?From:\s*([^\n]+).*?Subject:\s*([^\n]+)'
        match = re.search(gmail_pattern, body, re.DOTALL | re.IGNORECASE)
        if match:
            from_line = match.group(1).strip()
            subject_line = match.group(2).strip()
            
            email_match = re.search(r'<([^>]+)>|([^\s<>]+@[^\s<>]+)', from_line)
            if email_match:
                extracted['from'] = email_match.group(1) or email_match.group(2)
            
            extracted['subject'] = subject_line
            
            to_match = re.search(r'To:\s*([^\n]+)', body)
            if to_match:
                to_line = to_match.group(1).strip()
                to_email_match = re.search(r'<([^>]+)>|([^\s<>]+@[^\s<>]+)', to_line)
                if to_email_match:
                    extracted['to'] = to_email_match.group(1) or to_email_match.group(2)
            
            return extracted
        
        # Pattern 3: Outlook / Generic
        # From: email@domain.com
        # Sent: ...
        # To: email@domain.com
        # Subject: ...
        if 'From:' in body and ('Sent:' in body or 'Date:' in body):
            from_match = re.search(r'From:\s*([^\n]+)', body)
            subject_match = re.search(r'Subject:\s*([^\n]+)', body)
            to_match = re.search(r'To:\s*([^\n]+)', body)
            
            if from_match:
                from_line = from_match.group(1).strip()
                email_match = re.search(r'<([^>]+)>|([^\s<>]+@[^\s<>]+)', from_line)
                if email_match:
                    extracted['from'] = email_match.group(1) or email_match.group(2)
            
            if subject_match:
                extracted['subject'] = subject_match.group(1).strip()
            
            if to_match:
                to_line = to_match.group(1).strip()
                to_email_match = re.search(r'<([^>]+)>|([^\s<>]+@[^\s<>]+)', to_line)
                if to_email_match:
                    extracted['to'] = to_email_match.group(1) or to_email_match.group(2)
            
            if extracted:
                return extracted
        
        return None


class EmailPreprocessor:
    """
    Preprocesses emails before classification.
    Handles forwarded emails, mailing lists, and other special cases.
    """
    
    def __init__(self, rules: Optional[List[PreprocessingRule]] = None):
        """
        Initialize preprocessor with rules.
        
        Args:
            rules: List of preprocessing rules (sorted by priority)
        """
        self.rules = sorted(rules or [], key=lambda r: r.priority)
        logger.info(f"Initialized EmailPreprocessor with {len(self.rules)} rules")
    
    def preprocess(self, email: ProcessedEmail, raw_headers: Dict[str, str]) -> ProcessedEmail:
        """
        Apply preprocessing rules to email.
        
        Args:
            email: Email to preprocess
            raw_headers: Raw email headers (for accessing original headers)
            
        Returns:
            Preprocessed email (may be modified)
        """
        for rule in self.rules:
            if rule.matches(email, raw_headers):
                # Only log at debug level - will be shown in output if preprocessing actually changes something
                logger.debug(f"Applying preprocessing rule: {rule.name}")
                email = rule.apply(email, raw_headers)
                # Rules are applied in sequence (all matching rules apply)
        
        return email
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'EmailPreprocessor':
        """
        Load preprocessing rules from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration
            
        Returns:
            Configured EmailPreprocessor
        """
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        rules = []
        for rule_config in config.get('preprocessing_rules', []):
            rule = PreprocessingRule(
                name=rule_config['name'],
                description=rule_config.get('description', ''),
                priority=rule_config.get('priority', 100),
                match_from=rule_config.get('match_from'),
                match_to=rule_config.get('match_to'),
                match_subject=rule_config.get('match_subject'),
                extract_original_from=rule_config.get('extract_original_from'),
                extract_original_to=rule_config.get('extract_original_to'),
                extract_original_subject=rule_config.get('extract_original_subject'),
                parse_forwarded_body=rule_config.get('parse_forwarded_body', False),
                header_mappings=rule_config.get('header_mappings'),
                remove_prefix=rule_config.get('remove_prefix')
            )
            rules.append(rule)
        
        return cls(rules)
    
    def add_rule(self, rule: PreprocessingRule):
        """Add preprocessing rule and re-sort"""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority)
    
    def list_rules(self) -> List[Dict[str, Any]]:
        """Get list of rules"""
        return [
            {
                'name': rule.name,
                'description': rule.description,
                'priority': rule.priority,
                'transforms': {
                    'from': rule.extract_original_from is not None,
                    'to': rule.extract_original_to is not None,
                    'subject': rule.extract_original_subject is not None
                }
            }
            for rule in self.rules
        ]


def extract_headers_from_raw(raw_email: bytes) -> Dict[str, str]:
    """
    Extract all headers from raw email for preprocessing.
    
    Args:
        raw_email: Raw RFC822 email bytes
        
    Returns:
        Dictionary of all headers (as strings)
    """
    import email
    msg = email.message_from_bytes(raw_email)
    
    headers = {}
    for key, value in msg.items():
        # Store all headers (including X- headers)
        # Convert Header objects to strings
        if isinstance(value, str):
            headers[key] = value
        else:
            headers[key] = str(value)
    
    return headers

