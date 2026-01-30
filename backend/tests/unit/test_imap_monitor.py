"""
Test IMAP monitor functionality.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from backend.core.email.imap_monitor import IMAPMonitor, IMAPConfig


class TestIMAPConfig:
    """Test IMAP configuration"""
    
    def test_imap_config_creation(self):
        """Test creating IMAP config with defaults"""
        config = IMAPConfig(
            host="imap.gmail.com",
            username="test@gmail.com",
            password="secret"
        )
        
        assert config.host == "imap.gmail.com"
        assert config.username == "test@gmail.com"
        assert config.port == 993
        assert config.use_ssl == True
        assert config.folder == 'INBOX'
    
    def test_imap_config_custom_port(self):
        """Test IMAP config with custom port"""
        config = IMAPConfig(
            host="mail.example.com",
            username="user@example.com",
            password="pass",
            port=143,
            use_ssl=False
        )
        
        assert config.port == 143
        assert config.use_ssl == False


class TestIMAPMonitor:
    """Test IMAP monitor operations"""
    
    def test_initialization(self):
        """Test IMAPMonitor initialization"""
        config = IMAPConfig(
            host="imap.test.com",
            username="test@test.com",
            password="pass"
        )
        
        monitor = IMAPMonitor(config, timeout=60)
        
        assert monitor.config == config
        assert monitor.timeout == 60
        assert monitor.client is None
    
    def test_default_timeout(self):
        """Test default timeout is 30 seconds"""
        config = IMAPConfig(
            host="imap.test.com",
            username="test@test.com",
            password="pass"
        )
        
        monitor = IMAPMonitor(config)
        assert monitor.timeout == 30
    
    @patch('backend.core.email.imap_monitor.IMAPClient')
    def test_connect_success(self, mock_imap_client_class):
        """Test successful IMAP connection"""
        # Setup mock
        mock_client = MagicMock()
        mock_imap_client_class.return_value = mock_client
        
        config = IMAPConfig(
            host="imap.test.com",
            username="test@test.com",
            password="pass"
        )
        
        monitor = IMAPMonitor(config, timeout=30)
        result = monitor.connect()
        
        # Verify connection attempt
        mock_imap_client_class.assert_called_once_with(
            host="imap.test.com",
            port=993,
            ssl=True,
            timeout=30
        )
        
        # Verify login
        mock_client.login.assert_called_once_with("test@test.com", "pass")
        assert monitor.client == mock_client
        assert result == mock_client
    
    def test_mark_as_read(self, mock_imap_client):
        """Test marking email as read"""
        config = IMAPConfig(
            host="imap.test.com",
            username="test@test.com",
            password="pass"
        )
        
        monitor = IMAPMonitor(config)
        monitor.client = mock_imap_client
        
        result = monitor.mark_as_read("123")
        
        assert result == True
        mock_imap_client.add_flags.assert_called_with([123], ['\\Seen'])
    
    def test_move_to_folder(self, mock_imap_client):
        """Test moving email to folder"""
        config = IMAPConfig(
            host="imap.test.com",
            username="test@test.com",
            password="pass"
        )
        
        monitor = IMAPMonitor(config)
        monitor.client = mock_imap_client
        
        result = monitor.move_to_folder("456", "Archive")
        
        assert result == True
        mock_imap_client.copy.assert_called_with([456], "Archive")
        mock_imap_client.add_flags.assert_called_with([456], ['\\Deleted'])
        mock_imap_client.expunge.assert_called_once()
    
    def test_apply_color_label_red(self, mock_imap_client):
        """Test applying red color label (now just sets \\Flagged)"""
        config = IMAPConfig(
            host="imap.test.com",
            username="test@test.com",
            password="pass"
        )
        
        monitor = IMAPMonitor(config)
        monitor.client = mock_imap_client
        
        result = monitor.apply_color_label("789", 1)  # Red
        
        assert result == True
        # Now just sets \Flagged (color tracked for Phase 2 database)
        mock_imap_client.add_flags.assert_called_with([789], ['\\Flagged'])
    
    def test_apply_color_label_orange(self, mock_imap_client):
        """Test applying orange color label (now just sets \\Flagged)"""
        config = IMAPConfig(
            host="imap.test.com",
            username="test@test.com",
            password="pass"
        )
        
        monitor = IMAPMonitor(config)
        monitor.client = mock_imap_client
        
        result = monitor.apply_color_label("789", 2)  # Orange
        
        assert result == True
        # Now just sets \Flagged (color tracked for Phase 2 database)
        mock_imap_client.add_flags.assert_called_with([789], ['\\Flagged'])
    
    def test_apply_invalid_color(self, mock_imap_client):
        """Test applying invalid color code"""
        config = IMAPConfig(
            host="imap.test.com",
            username="test@test.com",
            password="pass"
        )
        
        monitor = IMAPMonitor(config)
        monitor.client = mock_imap_client
        
        result = monitor.apply_color_label("789", 99)  # Invalid
        
        assert result == False
        mock_imap_client.add_flags.assert_not_called()
    
    def test_get_folder_list(self, mock_imap_client):
        """Test getting folder list"""
        config = IMAPConfig(
            host="imap.test.com",
            username="test@test.com",
            password="pass"
        )
        
        monitor = IMAPMonitor(config)
        monitor.client = mock_imap_client
        
        folders = monitor.get_folder_list()
        
        assert len(folders) == 4
        assert 'INBOX' in folders
        assert 'Sent' in folders
        assert 'Trash' in folders
    
    def test_add_custom_flag(self, mock_imap_client):
        """Test adding custom IMAP flag"""
        config = IMAPConfig(
            host="imap.test.com",
            username="test@test.com",
            password="pass"
        )
        
        monitor = IMAPMonitor(config)
        monitor.client = mock_imap_client
        
        result = monitor.add_custom_flag("123", "Important")
        
        assert result == True
        # Should add $ prefix
        mock_imap_client.add_flags.assert_called_with([123], ['$Important'])
    
    def test_add_custom_flag_with_dollar(self, mock_imap_client):
        """Test adding custom flag that already has $ prefix"""
        config = IMAPConfig(
            host="imap.test.com",
            username="test@test.com",
            password="pass"
        )
        
        monitor = IMAPMonitor(config)
        monitor.client = mock_imap_client
        
        result = monitor.add_custom_flag("123", "$AlreadyHasDollar")
        
        assert result == True
        # Should not double-add $
        mock_imap_client.add_flags.assert_called_with([123], ['$AlreadyHasDollar'])
    
    def test_context_manager(self, mock_imap_client):
        """Test context manager connect/disconnect"""
        config = IMAPConfig(
            host="imap.test.com",
            username="test@test.com",
            password="pass"
        )
        
        with patch('backend.core.email.imap_monitor.IMAPClient') as mock_client_class:
            mock_client_class.return_value = mock_imap_client
            
            with IMAPMonitor(config) as monitor:
                assert monitor.client is not None
                mock_client_class.assert_called_once()
            
            # Should disconnect after context
            mock_imap_client.logout.assert_called_once()
    
    def test_operation_without_connection(self):
        """Test that operations fail gracefully without connection"""
        config = IMAPConfig(
            host="imap.test.com",
            username="test@test.com",
            password="pass"
        )
        
        monitor = IMAPMonitor(config)
        # No connection established
        
        with pytest.raises(RuntimeError, match="Not connected"):
            monitor.mark_as_read("123")
        
        with pytest.raises(RuntimeError, match="Not connected"):
            monitor.move_to_folder("123", "Archive")
        
        with pytest.raises(RuntimeError, match="Not connected"):
            monitor.apply_color_label("123", 1)

