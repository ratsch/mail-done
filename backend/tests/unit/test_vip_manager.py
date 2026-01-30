"""
Tests for VIP Manager
"""
import pytest
from pathlib import Path
from backend.core.email.vip_manager import VIPManager, VIPInfo
from backend.core.email.models import ProcessedEmail
from datetime import datetime


class TestVIPManager:
    """Test VIP sender management"""
    
    def test_vip_manager_initialization(self):
        """Test VIP manager initializes with empty config"""
        # Should not crash with missing config
        vip_manager = VIPManager(config_path="nonexistent.yaml")
        assert vip_manager.vips == {'urgent': set(), 'high': set(), 'medium': set()}
    
    def test_vip_manager_loads_config(self):
        """Test VIP manager loads configuration from YAML"""
        # Use the actual config file
        vip_manager = VIPManager(config_path="config/vip_senders.yaml")
        
        # Should have structure ready
        assert 'urgent' in vip_manager.vips
        assert 'high' in vip_manager.vips
        assert 'medium' in vip_manager.vips
    
    def test_add_vip_sender(self):
        """Test adding VIP sender programmatically"""
        vip_manager = VIPManager(config_path="nonexistent.yaml")
        
        vip_manager.add_vip("important@company.com", "urgent")
        
        assert vip_manager.is_vip("important@company.com")
        assert vip_manager.get_vip_level("important@company.com") == "urgent"
    
    def test_vip_detection_case_insensitive(self):
        """Test VIP detection is case-insensitive"""
        vip_manager = VIPManager(config_path="nonexistent.yaml")
        
        vip_manager.add_vip("Boss@Company.COM", "urgent")
        
        assert vip_manager.is_vip("boss@company.com")
        assert vip_manager.is_vip("BOSS@COMPANY.COM")
        assert vip_manager.get_vip_level("boss@company.com") == "urgent"
    
    def test_check_vip_with_processed_email(self):
        """Test VIP check with ProcessedEmail"""
        vip_manager = VIPManager(config_path="nonexistent.yaml")
        vip_manager.add_vip("vip@company.com", "high")
        
        # Create processed email from VIP
        email = ProcessedEmail(
            uid="1",
            message_id="<123>",
            subject="Important Email",
            from_address="vip@company.com",
            to_addresses=["me@example.com"],
            to_names=[],
            date=datetime.now(),
            body_markdown="Content",
            sender_domain="company.com"
        )
        
        vip_info = vip_manager.check_vip(email)
        
        assert vip_info is not None
        assert vip_info.sender == "vip@company.com"
        assert vip_info.level == "high"
        assert vip_info.color == 2  # Orange for high
        assert vip_info.priority == 2
    
    def test_check_non_vip_email(self):
        """Test VIP check returns None for non-VIP"""
        vip_manager = VIPManager(config_path="nonexistent.yaml")
        vip_manager.add_vip("vip@company.com", "urgent")
        
        email = ProcessedEmail(
            uid="1",
            message_id="<123>",
            subject="Regular Email",
            from_address="regular@example.com",
            to_addresses=["me@example.com"],
            to_names=[],
            date=datetime.now(),
            body_markdown="Content",
            sender_domain="example.com"
        )
        
        vip_info = vip_manager.check_vip(email)
        
        assert vip_info is None
    
    def test_vip_levels_have_correct_colors(self):
        """Test VIP levels map to correct Apple Mail colors"""
        vip_manager = VIPManager(config_path="nonexistent.yaml")
        
        vip_manager.add_vip("urgent@company.com", "urgent")
        vip_manager.add_vip("high@company.com", "high")
        vip_manager.add_vip("medium@company.com", "medium")
        
        # Create emails and check colors
        urgent_email = ProcessedEmail(
            uid="1", message_id="<1>", subject="Urgent", from_address="urgent@company.com",
            to_addresses=["me@example.com"], to_names=[], date=datetime.now(),
            body_markdown="", sender_domain="company.com"
        )
        high_email = ProcessedEmail(
            uid="2", message_id="<2>", subject="High", from_address="high@company.com",
            to_addresses=["me@example.com"], to_names=[], date=datetime.now(),
            body_markdown="", sender_domain="company.com"
        )
        medium_email = ProcessedEmail(
            uid="3", message_id="<3>", subject="Medium", from_address="medium@company.com",
            to_addresses=["me@example.com"], to_names=[], date=datetime.now(),
            body_markdown="", sender_domain="company.com"
        )
        
        assert vip_manager.check_vip(urgent_email).color == 1  # Red
        assert vip_manager.check_vip(high_email).color == 2  # Orange
        assert vip_manager.check_vip(medium_email).color == 3  # Yellow
    
    def test_get_vip_rules(self):
        """Test generation of classification rules from VIP config"""
        vip_manager = VIPManager(config_path="nonexistent.yaml")
        
        vip_manager.add_vip("urgent1@company.com", "urgent")
        vip_manager.add_vip("urgent2@company.com", "urgent")
        vip_manager.add_vip("high1@company.com", "high")
        
        rules = vip_manager.get_vip_rules()
        
        # Should generate 2 rules (urgent and high levels)
        assert len(rules) == 2
        
        # Check urgent rule
        urgent_rule = next(r for r in rules if r.name == "VIP - URGENT Priority")
        assert urgent_rule.priority == 1
        assert urgent_rule.action.color == 1
        assert 'VIP' in urgent_rule.action.labels
        assert 'URGENT' in urgent_rule.action.labels
    
    def test_remove_vip_sender(self):
        """Test removing VIP sender"""
        vip_manager = VIPManager(config_path="nonexistent.yaml")
        
        vip_manager.add_vip("temp@company.com", "urgent")
        assert vip_manager.is_vip("temp@company.com")
        
        vip_manager.remove_vip("temp@company.com")
        assert not vip_manager.is_vip("temp@company.com")
    
    def test_get_vip_stats(self):
        """Test VIP statistics"""
        vip_manager = VIPManager(config_path="nonexistent.yaml")
        
        vip_manager.add_vip("urgent@company.com", "urgent")
        vip_manager.add_vip("high1@company.com", "high")
        vip_manager.add_vip("high2@company.com", "high")
        vip_manager.add_vip("medium1@company.com", "medium")
        vip_manager.add_vip("medium2@company.com", "medium")
        vip_manager.add_vip("medium3@company.com", "medium")
        
        stats = vip_manager.get_vip_stats()
        
        assert stats['urgent'] == 1
        assert stats['high'] == 2
        assert stats['medium'] == 3

