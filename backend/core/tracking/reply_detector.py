"""
Reply Detection Logic

Analyzes emails to determine if they need replies using:
- Pattern matching (questions, action requests)
- AI metadata (needs_reply flag)
- Category analysis (applications, invitations require responses)
- Deadline extraction
"""
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List
from dateutil import parser as date_parser


@dataclass
class ReplyAnalysis:
    """Result of reply need analysis"""
    needs_reply: bool
    priority: int  # 1-10
    deadline: Optional[datetime]
    reason: str
    detected_by: str  # ai/pattern/category/vip
    question_detected: bool
    action_requested: bool
    deadline_source: Optional[str] = None  # Where deadline was found


class ReplyDetector:
    """Detect if email needs reply using multiple signals."""
    
    # Question patterns
    QUESTION_PATTERNS = [
        r'\?',  # Any question mark
        r'(?i)let me know',
        r'(?i)please (confirm|reply|respond|advise|let us know)',
        r'(?i)could you',
        r'(?i)would you',
        r'(?i)can you',
        r'(?i)are you (available|interested)',
        r'(?i)RSVP',
        r'(?i)looking forward to (hearing|your response)',
        r'(?i)await(ing)? your',
        r'(?i)I hope to hear',
        r'(?i)kindly (confirm|respond|reply)',
    ]
    
    # Action request patterns
    ACTION_PATTERNS = [
        r'(?i)please (send|provide|share|review|check|confirm)',
        r'(?i)need your (input|feedback|approval|response)',
        r'(?i)waiting for your',
        r'(?i)require your',
        r'(?i)could you (please )?(send|provide|share)',
        r'(?i)would (appreciate|like) (if you could|your)',
        r'(?i)action (required|needed)',
    ]
    
    # Deadline patterns
    DEADLINE_PATTERNS = [
        # Explicit deadline phrases
        (r'(?i)(?:reply|respond|RSVP|confirm).*?(?:by|before|until)\s+([A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})', 'reply_by'),
        (r'(?i)deadline[:\s]+([A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})', 'deadline'),
        (r'(?i)(?:by|before|until)\s+([A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})', 'by_date'),
        
        # Numeric dates
        (r'(?i)(?:reply|respond|RSVP|confirm).*?(?:by|before)\s+(\d{1,2}/\d{1,2}/\d{2,4})', 'reply_by_numeric'),
        (r'(?i)deadline[:\s]+(\d{1,2}/\d{1,2}/\d{2,4})', 'deadline_numeric'),
        
        # Relative dates
        (r'(?i)(?:reply|respond|confirm).*?(?:within|in)\s+(\d+)\s+(day|week|hour)s?', 'relative'),
        (r'(?i)(?:by|before)\s+(tomorrow|next week|end of week|Monday|Tuesday|Wednesday|Thursday|Friday)', 'relative_word'),
    ]
    
    # Categories that typically need replies
    REPLY_REQUIRED_CATEGORIES = {
        'application-phd': {'priority': 8, 'reason': 'PhD applications expect acknowledgment'},
        'application-postdoc': {'priority': 8, 'reason': 'Postdoc applications expect acknowledgment'},
        'application-intern': {'priority': 7, 'reason': 'Internship applications expect reply'},
        'application-bsc-msc-thesis': {'priority': 7, 'reason': 'Thesis applications expect reply'},
        'application-visiting': {'priority': 7, 'reason': 'Visiting applications expect reply'},
        
        'invitation-speaking': {'priority': 8, 'reason': 'Speaking invitations require RSVP'},
        'invitation-committee': {'priority': 9, 'reason': 'Committee invitations require response'},
        'invitation-grant': {'priority': 9, 'reason': 'Grant invitations require decision'},
        'invitation-editorial': {'priority': 9, 'reason': 'Editorial invitations require response'},
        'invitation-advisory': {'priority': 8, 'reason': 'Advisory invitations require decision'},
        'invitation-event': {'priority': 6, 'reason': 'Event invitations typically need RSVP'},
        'invitation-collaboration': {'priority': 8, 'reason': 'Collaboration invitations expect response'},
        
        'review-peer-journal': {'priority': 7, 'reason': 'Review requests expect accept/decline'},
        'review-peer-conference': {'priority': 7, 'reason': 'Review requests expect response'},
        'review-grant': {'priority': 9, 'reason': 'Grant reviews require commitment'},
        'review-phd-committee': {'priority': 8, 'reason': 'PhD committees require confirmation'},
        'review-hiring': {'priority': 8, 'reason': 'Hiring reviews expect response'},
        'review-promotion': {'priority': 9, 'reason': 'Promotion reviews require response'},
        
        'work-urgent': {'priority': 10, 'reason': 'Urgent work emails need immediate response'},
        'work-colleague': {'priority': 7, 'reason': 'Colleague emails typically need reply'},
        'work-student': {'priority': 6, 'reason': 'Student emails expect response'},
    }
    
    def __init__(self):
        # Compile regex patterns for efficiency
        self.question_regex = [re.compile(p) for p in self.QUESTION_PATTERNS]
        self.action_regex = [re.compile(p) for p in self.ACTION_PATTERNS]
        self.deadline_regex = [(re.compile(p), source) for p, source in self.DEADLINE_PATTERNS]
    
    def analyze(
        self,
        subject: str,
        body: str,
        ai_metadata: dict,
        vip_level: Optional[str] = None,
        category: Optional[str] = None
    ) -> ReplyAnalysis:
        """
        Comprehensive analysis to determine if email needs reply.
        
        Args:
            subject: Email subject
            body: Email body (markdown or text)
            ai_metadata: AI classification metadata (needs_reply, urgency, etc.)
            vip_level: VIP level if applicable (urgent/high/medium)
            category: AI category
        
        Returns:
            ReplyAnalysis with needs_reply, priority, deadline, reason
        """
        # Start with detection signals
        ai_says_reply = ai_metadata.get('needs_reply', False)
        question_detected = self.detect_questions(body)
        action_requested = self.detect_action_requests(body)
        deadline = self.extract_deadline(subject + '\n' + body)
        category_requires = category in self.REPLY_REQUIRED_CATEGORIES
        
        # Determine if reply is needed
        needs_reply = (
            ai_says_reply or 
            question_detected or 
            action_requested or 
            category_requires or
            vip_level in ['urgent', 'high']  # Always reply to high VIPs
        )
        
        # Calculate priority
        priority = self.calculate_priority(
            vip_level=vip_level,
            urgency=ai_metadata.get('ai_urgency', 'normal'),
            category=category,
            deadline=deadline,
            relevance_score=ai_metadata.get('relevance_score', 5),
            question_detected=question_detected,
            action_requested=action_requested
        )
        
        # Determine detection source and reason
        if vip_level == 'urgent':
            detected_by = 'vip'
            reason = f'VIP urgent email from {vip_level} sender'
        elif category_requires:
            detected_by = 'category'
            reason = self.REPLY_REQUIRED_CATEGORIES[category]['reason']
        elif ai_says_reply:
            detected_by = 'ai'
            reason = 'AI detected reply needed'
        elif question_detected:
            detected_by = 'pattern'
            reason = 'Questions detected in email'
        elif action_requested:
            detected_by = 'pattern'
            reason = 'Action items requested'
        else:
            detected_by = 'none'
            reason = 'No reply needed'
        
        return ReplyAnalysis(
            needs_reply=needs_reply,
            priority=priority,
            deadline=deadline,
            reason=reason,
            detected_by=detected_by,
            question_detected=question_detected,
            action_requested=action_requested,
            deadline_source=ai_metadata.get('deadline_source') if deadline else None
        )
    
    def detect_questions(self, body: str) -> bool:
        """Detect questions in email body."""
        if not body:
            return False
        
        # Check all question patterns
        for pattern in self.question_regex:
            if pattern.search(body):
                return True
        
        return False
    
    def detect_action_requests(self, body: str) -> bool:
        """Detect action items requested."""
        if not body:
            return False
        
        # Check all action patterns
        for pattern in self.action_regex:
            if pattern.search(body):
                return True
        
        return False
    
    def extract_deadline(self, text: str) -> Optional[datetime]:
        """
        Extract reply deadline from email text.
        
        Looks for patterns like:
        - "please reply by December 15, 2025"
        - "RSVP by 12/15/2025"
        - "deadline: December 15"
        - "respond within 3 days"
        """
        if not text:
            return None
        
        # Try each deadline pattern
        for pattern, source in self.deadline_regex:
            match = pattern.search(text)
            if match:
                try:
                    date_str = match.group(1)
                    
                    # Handle relative dates
                    if source == 'relative':
                        number = int(match.group(1))
                        unit = match.group(2).lower()
                        if unit.startswith('day'):
                            return datetime.now() + timedelta(days=number)
                        elif unit.startswith('week'):
                            return datetime.now() + timedelta(weeks=number)
                        elif unit.startswith('hour'):
                            return datetime.now() + timedelta(hours=number)
                    
                    elif source == 'relative_word':
                        word = date_str.lower()
                        if word == 'tomorrow':
                            return datetime.now() + timedelta(days=1)
                        elif word == 'next week':
                            return datetime.now() + timedelta(weeks=1)
                        elif word == 'end of week':
                            # Find next Friday
                            days_ahead = 4 - datetime.now().weekday()
                            if days_ahead <= 0:
                                days_ahead += 7
                            return datetime.now() + timedelta(days=days_ahead)
                        elif word in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']:
                            # Find next occurrence of this weekday
                            weekdays = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 
                                       'thursday': 3, 'friday': 4}
                            target_day = weekdays[word]
                            days_ahead = target_day - datetime.now().weekday()
                            if days_ahead <= 0:
                                days_ahead += 7
                            return datetime.now() + timedelta(days=days_ahead)
                    
                    else:
                        # Parse absolute dates
                        parsed_date = date_parser.parse(date_str, fuzzy=True)
                        
                        # If no year specified and date is in past, assume next year
                        if parsed_date.year == datetime.now().year and parsed_date < datetime.now():
                            parsed_date = parsed_date.replace(year=parsed_date.year + 1)
                        
                        return parsed_date
                        
                except (ValueError, AttributeError):
                    continue
        
        return None
    
    def calculate_priority(
        self,
        vip_level: Optional[str],
        urgency: str,
        category: Optional[str],
        deadline: Optional[datetime],
        relevance_score: int,
        question_detected: bool,
        action_requested: bool
    ) -> int:
        """
        Calculate reply priority (1-10).
        
        Scoring factors:
        - VIP level: urgent=10, high=9, medium=8
        - Deadline proximity: <24h=+2, <48h=+1, <week=+0.5
        - Category base priority
        - Urgency: urgent=+1, normal=0, low=-1
        - Relevance: >8=+1, >6=+0.5
        - Questions/actions: +0.5 each
        """
        # Start with base priority from category
        if category and category in self.REPLY_REQUIRED_CATEGORIES:
            priority = self.REPLY_REQUIRED_CATEGORIES[category]['priority']
        else:
            priority = 5  # Default medium priority
        
        # VIP override (highest priority)
        if vip_level == 'urgent':
            priority = max(priority, 10)
        elif vip_level == 'high':
            priority = max(priority, 9)
        elif vip_level == 'medium':
            priority = max(priority, 8)
        
        # Deadline proximity boost
        if deadline:
            time_until = deadline - datetime.now()
            if time_until.total_seconds() > 0:  # Only if deadline hasn't passed
                hours_until = time_until.total_seconds() / 3600
                if hours_until < 24:
                    priority = min(10, priority + 2)
                elif hours_until < 48:
                    priority = min(10, priority + 1)
                elif hours_until < 168:  # 1 week
                    priority = min(10, priority + 0.5)
            else:
                # Overdue - max priority
                priority = 10
        
        # Urgency adjustment
        if urgency == 'urgent':
            priority = min(10, priority + 1)
        elif urgency == 'low':
            priority = max(1, priority - 1)
        
        # Relevance boost
        if relevance_score >= 8:
            priority = min(10, priority + 1)
        elif relevance_score >= 6:
            priority = min(10, priority + 0.5)
        
        # Pattern detection bonus
        if question_detected:
            priority = min(10, priority + 0.5)
        if action_requested:
            priority = min(10, priority + 0.5)
        
        return round(priority)

