"""
Unit Tests for Reply Detector

Tests the smart detection of emails that need replies.
"""
import pytest
from datetime import datetime, timedelta

from backend.core.tracking.reply_detector import ReplyDetector, ReplyAnalysis


class TestReplyDetector:
    """Test suite for ReplyDetector"""
    
    @pytest.fixture
    def detector(self):
        """Create reply detector instance"""
        return ReplyDetector()
    
    # =========================================================================
    # Question Detection Tests
    # =========================================================================
    
    def test_detect_simple_question(self, detector):
        """Test detection of simple question marks"""
        body = "Are you available for a meeting next week?"
        
        assert detector.detect_questions(body) is True
    
    def test_detect_let_me_know(self, detector):
        """Test 'let me know' pattern"""
        body = "Please let me know if you're interested."
        
        assert detector.detect_questions(body) is True
    
    def test_detect_please_confirm(self, detector):
        """Test 'please confirm' pattern"""
        body = "Please confirm your attendance by Friday."
        
        assert detector.detect_questions(body) is True
    
    def test_detect_rsvp(self, detector):
        """Test RSVP detection"""
        body = "RSVP by December 15th."
        
        assert detector.detect_questions(body) is True
    
    def test_no_question_statement(self, detector):
        """Test that statements don't trigger false positives"""
        body = "I'm writing to inform you about the conference."
        
        assert detector.detect_questions(body) is False
    
    # =========================================================================
    # Action Request Detection Tests
    # =========================================================================
    
    def test_detect_please_send(self, detector):
        """Test 'please send' action request"""
        body = "Please send me your CV and research statement."
        
        assert detector.detect_action_requests(body) is True
    
    def test_detect_need_your_input(self, detector):
        """Test 'need your input' pattern"""
        body = "We need your input on the proposal."
        
        assert detector.detect_action_requests(body) is True
    
    def test_detect_waiting_for(self, detector):
        """Test 'waiting for' pattern"""
        body = "We're waiting for your decision on this matter."
        
        assert detector.detect_action_requests(body) is True
    
    # =========================================================================
    # Deadline Extraction Tests
    # =========================================================================
    
    def test_extract_explicit_deadline(self, detector):
        """Test extraction of explicit deadline"""
        text = "Please reply by December 15, 2025."
        
        deadline = detector.extract_deadline(text)
        
        assert deadline is not None
        assert deadline.month == 12
        assert deadline.day == 15
        assert deadline.year == 2025
    
    def test_extract_deadline_format(self, detector):
        """Test numeric date format"""
        text = "RSVP by 12/15/2025"
        
        deadline = detector.extract_deadline(text)
        
        assert deadline is not None
        assert deadline.month == 12
        assert deadline.day == 15
    
    def test_extract_relative_deadline(self, detector):
        """Test relative deadline (within 3 days)"""
        text = "Please respond within 3 days."
        
        deadline = detector.extract_deadline(text)
        
        assert deadline is not None
        # Should be ~3 days from now
        expected = datetime.now() + timedelta(days=3)
        assert abs((deadline - expected).days) <= 1
    
    def test_extract_tomorrow_deadline(self, detector):
        """Test 'tomorrow' pattern"""
        text = "Reply by tomorrow please."
        
        deadline = detector.extract_deadline(text)
        
        assert deadline is not None
        expected = datetime.now() + timedelta(days=1)
        assert abs((deadline - expected).days) <= 1
    
    def test_no_deadline_in_text(self, detector):
        """Test no deadline present"""
        text = "Just wanted to touch base."
        
        deadline = detector.extract_deadline(text)
        
        assert deadline is None
    
    # =========================================================================
    # Priority Calculation Tests
    # =========================================================================
    
    def test_priority_vip_urgent(self, detector):
        """Test VIP urgent gets priority 10"""
        priority = detector.calculate_priority(
            vip_level='urgent',
            urgency='normal',
            category='work-colleague',
            deadline=None,
            relevance_score=5,
            question_detected=False,
            action_requested=False
        )
        
        assert priority == 10
    
    def test_priority_vip_high(self, detector):
        """Test VIP high gets priority 9"""
        priority = detector.calculate_priority(
            vip_level='high',
            urgency='normal',
            category='work-colleague',
            deadline=None,
            relevance_score=5,
            question_detected=False,
            action_requested=False
        )
        
        assert priority == 9
    
    def test_priority_urgent_deadline(self, detector):
        """Test urgent deadline (<24h) boosts priority"""
        deadline = datetime.now() + timedelta(hours=12)
        
        priority = detector.calculate_priority(
            vip_level=None,
            urgency='normal',
            category='application-phd',
            deadline=deadline,
            relevance_score=5,
            question_detected=False,
            action_requested=False
        )
        
        # Base: 8 (application-phd) + 2 (deadline <24h) = 10
        assert priority >= 9
    
    def test_priority_category_base(self, detector):
        """Test category-based priority"""
        priority = detector.calculate_priority(
            vip_level=None,
            urgency='normal',
            category='invitation-grant',
            deadline=None,
            relevance_score=5,
            question_detected=False,
            action_requested=False
        )
        
        # invitation-grant has base priority 9
        assert priority == 9
    
    def test_priority_high_relevance_boost(self, detector):
        """Test high relevance score boosts priority"""
        priority = detector.calculate_priority(
            vip_level=None,
            urgency='normal',
            category='application-phd',
            deadline=None,
            relevance_score=9,  # High relevance
            question_detected=False,
            action_requested=False
        )
        
        # Base: 8 + 1 (relevance >8) = 9
        assert priority == 9
    
    def test_priority_question_boost(self, detector):
        """Test question detection boosts priority"""
        priority = detector.calculate_priority(
            vip_level=None,
            urgency='normal',
            category='work-colleague',
            deadline=None,
            relevance_score=5,
            question_detected=True,
            action_requested=False
        )
        
        # Base: 7 + 0.5 (question) = 8 (rounded)
        assert priority == 8
    
    # =========================================================================
    # Full Analysis Tests
    # =========================================================================
    
    def test_analyze_phd_application(self, detector):
        """Test full analysis of PhD application"""
        subject = "PhD Application - Machine Learning"
        body = "Dear Professor, I am interested in joining your lab. Could you let me know if you have openings?"
        ai_metadata = {
            'needs_reply': True,
            'ai_urgency': 'normal',
            'relevance_score': 8
        }
        
        analysis = detector.analyze(
            subject=subject,
            body=body,
            ai_metadata=ai_metadata,
            vip_level=None,
            category='application-phd'
        )
        
        assert analysis.needs_reply is True
        assert analysis.priority >= 7  # application-phd base
        assert analysis.question_detected is True
        assert analysis.detected_by in ['category', 'ai', 'pattern']
    
    def test_analyze_vip_urgent(self, detector):
        """Test VIP urgent email"""
        subject = "Urgent: Meeting tomorrow"
        body = "Can you attend the meeting tomorrow at 2pm?"
        ai_metadata = {
            'needs_reply': False,  # AI might miss it
            'ai_urgency': 'urgent'
        }
        
        analysis = detector.analyze(
            subject=subject,
            body=body,
            ai_metadata=ai_metadata,
            vip_level='urgent',
            category='work-urgent'
        )
        
        assert analysis.needs_reply is True
        assert analysis.priority == 10  # VIP urgent
        assert analysis.detected_by == 'vip'
    
    def test_analyze_no_reply_needed(self, detector):
        """Test newsletter doesn't need reply"""
        subject = "Weekly Newsletter - Nature"
        body = "Here are this week's top research papers..."
        ai_metadata = {
            'needs_reply': False,
            'ai_urgency': 'low'
        }
        
        analysis = detector.analyze(
            subject=subject,
            body=body,
            ai_metadata=ai_metadata,
            vip_level=None,
            category='newsletter-scientific'
        )
        
        assert analysis.needs_reply is False
        assert analysis.detected_by == 'none'
    
    def test_analyze_invitation_with_deadline(self, detector):
        """Test speaking invitation with RSVP deadline"""
        subject = "Invitation to speak at ICML 2026"
        body = "We would like to invite you to give a keynote. Please RSVP by March 1, 2026."
        ai_metadata = {
            'needs_reply': True,
            'ai_urgency': 'normal',
            'relevance_score': 8
        }
        
        analysis = detector.analyze(
            subject=subject,
            body=body,
            ai_metadata=ai_metadata,
            vip_level=None,
            category='invitation-speaking'
        )
        
        assert analysis.needs_reply is True
        assert analysis.deadline is not None
        assert analysis.deadline.month == 3
        assert analysis.deadline.day == 1
        assert analysis.priority >= 8  # invitation-speaking base + relevance

