"""
Test rule-based classification system.
"""
import pytest
from datetime import datetime
from backend.core.email.rule_classifier import (
    RuleCondition,
    ClassificationRule,
    RuleBasedClassifier
)
from backend.core.email.models import (
    ProcessedEmail,
    EmailCategory,
    EmailAction,
    AppleMailColor
)


class TestRuleCondition:
    """Test individual rule conditions"""
    
    @pytest.fixture
    def sample_email(self):
        return ProcessedEmail(
            uid="123",
            message_id="<test@example.com>",
            subject="Newsletter: Weekly Updates",
            from_address="newsletter@genomeweb.com",
            to_addresses=["me@example.com"],
            date=datetime.now(),
            body_markdown="Content",
            sender_domain="genomeweb.com",
            attachment_info=[]
        )
    
    def test_condition_equals(self, sample_email):
        """Test exact match condition"""
        condition = RuleCondition(
            field='from',
            pattern='newsletter@genomeweb.com',
            match_type='equals'
        )
        
        assert condition.evaluate(sample_email) == True
        
        # Different email should not match
        sample_email.from_address = "other@example.com"
        assert condition.evaluate(sample_email) == False
    
    def test_condition_contains(self, sample_email):
        """Test substring match"""
        condition = RuleCondition(
            field='subject',
            pattern='newsletter',
            match_type='contains',
            case_sensitive=False
        )
        
        assert condition.evaluate(sample_email) == True
        
        # Case insensitive
        sample_email.subject = "NEWSLETTER: Updates"
        assert condition.evaluate(sample_email) == True
    
    def test_condition_regex_matches(self, sample_email):
        """Test regex pattern matching"""
        condition = RuleCondition(
            field='domain',
            pattern=r'genomeweb\.com$',
            match_type='matches'
        )
        
        assert condition.evaluate(sample_email) == True
        
        # Subdomain should not match with $ anchor
        sample_email.sender_domain = "mail.genomeweb.com"
        assert condition.evaluate(sample_email) == True  # Still matches
        
        # Different domain
        sample_email.sender_domain = "example.org"
        assert condition.evaluate(sample_email) == False
    
    def test_condition_case_sensitive(self, sample_email):
        """Test case sensitive matching"""
        condition = RuleCondition(
            field='subject',
            pattern='Newsletter',
            match_type='contains',
            case_sensitive=True
        )
        
        assert condition.evaluate(sample_email) == True
        
        # All lowercase should not match
        sample_email.subject = "newsletter: weekly"
        assert condition.evaluate(sample_email) == False
    
    def test_condition_to_field(self, sample_email):
        """Test matching recipient addresses"""
        condition = RuleCondition(
            field='to',
            pattern='me@example.com',
            match_type='contains'
        )
        
        assert condition.evaluate(sample_email) == True
    
    def test_condition_invalid_regex(self, sample_email):
        """Test handling of invalid regex"""
        condition = RuleCondition(
            field='subject',
            pattern='[invalid(regex',  # Invalid regex
            match_type='matches'
        )
        
        # Should return False and log error
        assert condition.evaluate(sample_email) == False


class TestClassificationRule:
    """Test complete classification rules"""
    
    @pytest.fixture
    def receipt_email(self):
        return ProcessedEmail(
            uid="456",
            message_id="<receipt@example.com>",
            subject="Your Amazon Order #12345",
            from_address="auto-confirm@amazon.com",
            to_addresses=["me@example.com"],
            date=datetime.now(),
            body_markdown="Order confirmation",
            sender_domain="amazon.com",
            attachment_info=[]
        )
    
    def test_rule_with_single_condition(self, receipt_email):
        """Test rule with one condition"""
        rule = ClassificationRule(
            name="Amazon Receipts",
            conditions=[
                RuleCondition(field='domain', pattern='amazon.com', match_type='equals')
            ],
            category=EmailCategory.RECEIPT,
            action=EmailAction(type='move', folder='Receipts')
        )
        
        assert rule.matches(receipt_email) == True
    
    def test_rule_with_multiple_conditions_all_match(self, receipt_email):
        """Test rule with AND conditions (all must match)"""
        rule = ClassificationRule(
            name="Amazon Orders",
            conditions=[
                RuleCondition(field='domain', pattern='amazon.com', match_type='equals'),
                RuleCondition(field='subject', pattern='order', match_type='contains', case_sensitive=False)
            ],
            category=EmailCategory.RECEIPT,
            action=EmailAction(type='move', folder='Receipts')
        )
        
        # Both conditions match
        assert rule.matches(receipt_email) == True
    
    def test_rule_with_multiple_conditions_partial_match(self, receipt_email):
        """Test that partial match fails (need ALL conditions)"""
        rule = ClassificationRule(
            name="Test Rule",
            conditions=[
                RuleCondition(field='domain', pattern='amazon.com', match_type='equals'),
                RuleCondition(field='subject', pattern='invoice', match_type='contains')  # Won't match
            ],
            category=EmailCategory.RECEIPT,
            action=EmailAction(type='move', folder='Receipts')
        )
        
        # Only one condition matches - should fail
        assert rule.matches(receipt_email) == False
    
    def test_rule_priority(self):
        """Test rule priority ordering"""
        rule1 = ClassificationRule(
            name="Low priority",
            conditions=[RuleCondition(field='from', pattern='test', match_type='contains')],
            category=EmailCategory.WORK,
            action=EmailAction(type='keep'),
            priority=100
        )
        
        rule2 = ClassificationRule(
            name="High priority",
            conditions=[RuleCondition(field='from', pattern='urgent', match_type='contains')],
            category=EmailCategory.WORK,
            action=EmailAction(type='color', color=AppleMailColor.RED),
            priority=1
        )
        
        classifier = RuleBasedClassifier([rule1, rule2])
        
        # Should be sorted by priority
        assert classifier.rules[0].name == "High priority"
        assert classifier.rules[1].name == "Low priority"


class TestRuleBasedClassifier:
    """Test the complete classifier"""
    
    def test_classify_with_matching_rule(self):
        """Test classification when rule matches"""
        rules = [
            ClassificationRule(
                name="Newsletters",
                conditions=[
                    RuleCondition(field='subject', pattern='newsletter', match_type='contains')
                ],
                category=EmailCategory.NEWSLETTER,
                action=EmailAction(type='label', labels=['newsletter'])
            )
        ]
        
        classifier = RuleBasedClassifier(rules)
        
        email = ProcessedEmail(
            uid="1",
            message_id="<test@example.com>",
            subject="Monthly Newsletter",
            from_address="news@example.com",
            to_addresses=["me@example.com"],
            date=datetime.now(),
            body_markdown="News content",
            sender_domain="example.com",
            attachment_info=[]
        )
        
        result = classifier.classify(email)
        
        assert result is not None
        category, action, _ = result
        assert category == EmailCategory.NEWSLETTER
        assert action.type == 'label'
    
    def test_classify_with_no_matching_rule(self):
        """Test classification when no rules match"""
        rules = [
            ClassificationRule(
                name="Specific Rule",
                conditions=[
                    RuleCondition(field='from', pattern='specific@example.com', match_type='equals')
                ],
                category=EmailCategory.WORK,
                action=EmailAction(type='keep')
            )
        ]
        
        classifier = RuleBasedClassifier(rules)
        
        email = ProcessedEmail(
            uid="2",
            message_id="<test2@example.com>",
            subject="Random Email",
            from_address="other@example.com",
            to_addresses=["me@example.com"],
            date=datetime.now(),
            body_markdown="Content",
            sender_domain="example.com",
            attachment_info=[]
        )
        
        result = classifier.classify(email)
        
        # No match - should return None (fall through to AI)
        assert result is None
    
    def test_classify_priority_order(self):
        """Test that rules are evaluated in priority order"""
        rules = [
            ClassificationRule(
                name="Low priority - generic",
                conditions=[
                    RuleCondition(field='from', pattern='example.com', match_type='contains')
                ],
                category=EmailCategory.WORK,
                action=EmailAction(type='keep'),
                priority=100
            ),
            ClassificationRule(
                name="High priority - specific",
                conditions=[
                    RuleCondition(field='from', pattern='urgent@example.com', match_type='equals')
                ],
                category=EmailCategory.WORK,
                action=EmailAction(type='color', color=AppleMailColor.RED),
                priority=1
            )
        ]
        
        classifier = RuleBasedClassifier(rules)
        
        email = ProcessedEmail(
            uid="3",
            message_id="<test3@example.com>",
            subject="Test",
            from_address="urgent@example.com",
            to_addresses=["me@example.com"],
            date=datetime.now(),
            body_markdown="Content",
            sender_domain="example.com",
            attachment_info=[]
        )
        
        result = classifier.classify(email)
        
        # Should match high priority rule first
        category, action, _ = result
        assert action.type == 'color'  # High priority action
        assert action.color == AppleMailColor.RED
    
    def test_stop_on_match_behavior(self):
        """Test stop_on_match flag"""
        email = ProcessedEmail(
            uid="4",
            message_id="<test4@example.com>",
            subject="Urgent Newsletter",
            from_address="news@example.com",
            to_addresses=["me@example.com"],
            date=datetime.now(),
            body_markdown="Content",
            sender_domain="example.com",
            attachment_info=[]
        )
        
        # Rule that matches but doesn't stop
        rule1 = ClassificationRule(
            name="Newsletter check",
            conditions=[
                RuleCondition(field='subject', pattern='newsletter', match_type='contains')
            ],
            category=EmailCategory.NEWSLETTER,
            action=EmailAction(type='label', labels=['newsletter']),
            priority=10,
            stop_on_match=False  # Don't stop
        )
        
        # Second rule that also matches
        rule2 = ClassificationRule(
            name="Urgent check",
            conditions=[
                RuleCondition(field='subject', pattern='urgent', match_type='contains')
            ],
            category=EmailCategory.WORK,
            action=EmailAction(type='color', color=AppleMailColor.RED),
            priority=20,
            stop_on_match=True
        )
        
        classifier = RuleBasedClassifier([rule1, rule2])
        result = classifier.classify(email)
        
        # Should return first match (rule1) since it comes first by priority
        category, action, _ = result
        assert category == EmailCategory.NEWSLETTER
    
    def test_load_from_yaml(self):
        """Test loading rules from YAML file"""
        # This will test the actual config file
        import tempfile
        import yaml
        
        yaml_content = """
rules:
  - name: "Test Rule"
    priority: 10
    category: newsletter
    conditions:
      - field: subject
        pattern: "test"
        match_type: contains
    action:
      type: label
      labels: ["test"]
    stop_on_match: true
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            classifier = RuleBasedClassifier.from_yaml(f.name)
            
            assert len(classifier.rules) == 1
            assert classifier.rules[0].name == "Test Rule"
            assert classifier.rules[0].category == EmailCategory.NEWSLETTER
    
    def test_add_remove_rules_dynamically(self):
        """Test adding and removing rules at runtime"""
        classifier = RuleBasedClassifier([])
        
        assert len(classifier.rules) == 0
        
        # Add a rule
        rule = ClassificationRule(
            name="Dynamic Rule",
            conditions=[RuleCondition(field='from', pattern='test', match_type='contains')],
            category=EmailCategory.WORK,
            action=EmailAction(type='keep')
        )
        
        classifier.add_rule(rule)
        assert len(classifier.rules) == 1
        
        # Remove the rule
        removed = classifier.remove_rule("Dynamic Rule")
        assert removed == True
        assert len(classifier.rules) == 0
        
        # Try to remove non-existent rule
        removed = classifier.remove_rule("Nonexistent")
        assert removed == False


class TestRuleExamples:
    """Test example rules that would be common in practice"""
    
    def test_amazon_receipt_rule(self):
        """Test rule for Amazon receipts"""
        rule = ClassificationRule(
            name="Amazon Receipts",
            conditions=[
                RuleCondition(field='domain', pattern='amazon.com', match_type='equals'),
                RuleCondition(field='subject', pattern='order|shipment', match_type='matches')
            ],
            category=EmailCategory.RECEIPT,
            action=EmailAction(type='move', folder='Receipts')
        )
        
        email = ProcessedEmail(
            uid="1",
            message_id="<a@amazon.com>",
            subject="Your Amazon order has shipped",
            from_address="auto-confirm@amazon.com",
            to_addresses=["me@example.com"],
            date=datetime.now(),
            body_markdown="Shipment details",
            sender_domain="amazon.com",
            attachment_info=[]
        )
        
        assert rule.matches(email) == True
    
    def test_noreply_notification_rule(self):
        """Test rule for noreply notifications"""
        rule = ClassificationRule(
            name="NoReply Notifications",
            conditions=[
                RuleCondition(field='from', pattern='noreply@|no-reply@', match_type='matches')
            ],
            category=EmailCategory.NOTIFICATION,
            action=EmailAction(type='label', labels=['automated'])
        )
        
        # Test with noreply
        email1 = ProcessedEmail(
            uid="2",
            message_id="<n@example.com>",
            subject="Alert",
            from_address="noreply@service.com",
            to_addresses=["me@example.com"],
            date=datetime.now(),
            body_markdown="Alert message",
            sender_domain="service.com",
            attachment_info=[]
        )
        
        assert rule.matches(email1) == True
        
        # Test with no-reply
        email2 = ProcessedEmail(
            uid="3",
            message_id="<n2@example.com>",
            subject="Alert",
            from_address="no-reply@service.com",
            to_addresses=["me@example.com"],
            date=datetime.now(),
            body_markdown="Alert",
            sender_domain="service.com",
            attachment_info=[]
        )
        
        assert rule.matches(email2) == True
    
    def test_urgent_email_rule(self):
        """Test rule for urgent emails"""
        rule = ClassificationRule(
            name="Urgent Emails",
            conditions=[
                RuleCondition(field='subject', pattern='urgent|asap|emergency', match_type='matches', case_sensitive=False)
            ],
            category=EmailCategory.WORK,
            action=EmailAction(type='color', color=AppleMailColor.RED),
            priority=1,  # High priority
            stop_on_match=False  # Allow other rules to also apply
        )
        
        email = ProcessedEmail(
            uid="4",
            message_id="<u@example.com>",
            subject="URGENT: Project Deadline",
            from_address="colleague@work.com",
            to_addresses=["me@work.com"],
            date=datetime.now(),
            body_markdown="Need this ASAP",
            sender_domain="work.com",
            attachment_info=[]
        )
        
        assert rule.matches(email) == True
    
    def test_work_domain_rule(self):
        """Test rule for work domain"""
        rule = ClassificationRule(
            name="Institution Work Emails",
            conditions=[
                RuleCondition(field='domain', pattern='institution\\.edu$|cs\\.institution\\.edu$', match_type='matches')
            ],
            category=EmailCategory.WORK,
            action=EmailAction(type='keep')
        )
        
        # Test main domain
        email1 = ProcessedEmail(
            uid="5",
            message_id="<e1@institution.edu>",
            subject="Work stuff",
            from_address="colleague@institution.edu",
            to_addresses=["me@institution.edu"],
            date=datetime.now(),
            body_markdown="Content",
            sender_domain="institution.edu",
            attachment_info=[]
        )

        assert rule.matches(email1) == True

        # Test subdomain
        email2 = ProcessedEmail(
            uid="6",
            message_id="<e2@cs.institution.edu>",
            subject="Work stuff",
            from_address="prof@cs.institution.edu",
            to_addresses=["me@institution.edu"],
            date=datetime.now(),
            body_markdown="Content",
            sender_domain="cs.institution.edu",
            attachment_info=[]
        )

        assert rule.matches(email2) == True


class TestRuleBasedClassifierIntegration:
    """Test full classifier with multiple rules"""
    
    def test_complex_rule_set(self):
        """Test classifier with multiple prioritized rules"""
        rules = [
            # High priority: urgent
            ClassificationRule(
                name="Urgent",
                conditions=[
                    RuleCondition(field='subject', pattern='urgent', match_type='contains')
                ],
                category=EmailCategory.WORK,
                action=EmailAction(type='color', color=AppleMailColor.RED),
                priority=1
            ),
            # Medium priority: work domain
            ClassificationRule(
                name="Work",
                conditions=[
                    RuleCondition(field='domain', pattern='company.com', match_type='equals')
                ],
                category=EmailCategory.WORK,
                action=EmailAction(type='keep'),
                priority=50
            ),
            # Low priority: newsletter
            ClassificationRule(
                name="Newsletter",
                conditions=[
                    RuleCondition(field='subject', pattern='newsletter', match_type='contains')
                ],
                category=EmailCategory.NEWSLETTER,
                action=EmailAction(type='archive'),
                priority=100
            )
        ]
        
        classifier = RuleBasedClassifier(rules)
        
        # Test urgent email from work domain
        urgent_work = ProcessedEmail(
            uid="1",
            message_id="<uw@company.com>",
            subject="URGENT: Review needed",
            from_address="boss@company.com",
            to_addresses=["me@company.com"],
            date=datetime.now(),
            body_markdown="Please review ASAP",
            sender_domain="company.com",
            attachment_info=[]
        )
        
        result = classifier.classify(urgent_work)
        category, action, _ = result
        
        # Should match urgent rule (priority 1) not work rule (priority 50)
        assert category == EmailCategory.WORK
        assert action.color == AppleMailColor.RED
    
    def test_list_rules(self):
        """Test listing all rules"""
        rules = [
            ClassificationRule(
                name="Rule1",
                conditions=[RuleCondition(field='from', pattern='a', match_type='contains')],
                category=EmailCategory.WORK,
                action=EmailAction(type='keep'),
                priority=10
            ),
            ClassificationRule(
                name="Rule2",
                conditions=[RuleCondition(field='subject', pattern='b', match_type='contains')],
                category=EmailCategory.PERSONAL,
                action=EmailAction(type='color', color=AppleMailColor.ORANGE),
                priority=20
            )
        ]
        
        classifier = RuleBasedClassifier(rules)
        rule_list = classifier.list_rules()
        
        assert len(rule_list) == 2
        assert rule_list[0]['name'] == 'Rule1'
        assert rule_list[0]['priority'] == 10
        assert rule_list[1]['name'] == 'Rule2'

