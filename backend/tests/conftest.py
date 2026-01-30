"""
Test setup mirroring inbox-zero's testing approach.
Provides fixtures for email testing.
"""
# Load .env BEFORE any other imports (encryption module needs DB_ENCRYPTION_KEY at import time)
import os
from pathlib import Path
from dotenv import load_dotenv

# Load from repo root .env
_repo_root = Path(__file__).parent.parent.parent
load_dotenv(_repo_root / ".env")

# Generate test encryption key if not set
if not os.getenv("DB_ENCRYPTION_KEY"):
    from cryptography.fernet import Fernet
    os.environ["DB_ENCRYPTION_KEY"] = Fernet.generate_key().decode()

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
import json
from pathlib import Path
from datetime import datetime

from backend.core.email.models import (
    Email,
    ProcessedEmail,
    EmailCategory,
    SenderType,
    ReplyStatus,
    ClassificationResult,
    EmailAction,
)


@pytest.fixture
def sample_emails():
    """Sample test emails following inbox-zero patterns"""
    return {
        "personal_urgent": {
            "uid": "1",
            "message_id": "<test1@example.com>",
            "subject": "Emergency: Family matter",
            "from_address": "mom@family.com",
            "to_addresses": ["me@example.com"],
            "date": datetime.now(),
            "body_text": "Please call me urgently about...",
            "expected": {
                "category": EmailCategory.PERSONAL,
                "is_urgent": True,
                "action_type": "color",
            }
        },
        "work_meeting": {
            "uid": "2",
            "message_id": "<test2@example.com>",
            "subject": "Team meeting tomorrow at 10am",
            "from_address": "colleague@company.com",
            "to_addresses": ["me@company.com"],
            "date": datetime.now(),
            "body_text": "Hi team, reminder about our meeting...",
            "expected": {
                "category": EmailCategory.WORK,
                "is_urgent": False,
                "action_type": "keep",
            }
        },
        "receipt": {
            "uid": "3",
            "message_id": "<test3@example.com>",
            "subject": "Your order #12345",
            "from_address": "noreply@amazon.com",
            "to_addresses": ["me@example.com"],
            "date": datetime.now(),
            "body_text": "Thank you for your order...",
            "expected": {
                "category": EmailCategory.RECEIPT,
                "is_urgent": False,
                "action_type": "move",
            }
        },
        "newsletter": {
            "uid": "4",
            "message_id": "<test4@example.com>",
            "subject": "Weekly Newsletter - AI Updates",
            "from_address": "newsletter@techblog.com",
            "to_addresses": ["me@example.com"],
            "date": datetime.now(),
            "body_text": "This week in AI: Latest developments...",
            "expected": {
                "category": EmailCategory.NEWSLETTER,
                "is_urgent": False,
                "action_type": "label",
            }
        },
        "cold_email": {
            "uid": "5",
            "message_id": "<test5@example.com>",
            "subject": "Grow your business 10x",
            "from_address": "sales@randomcompany.com",
            "to_addresses": ["me@example.com"],
            "date": datetime.now(),
            "body_text": "Hi, I noticed your company and wanted to reach out...",
            "expected": {
                "category": EmailCategory.COLD_EMAIL,
                "sender_type": SenderType.COLD_EMAIL,
                "action_type": "archive",
            }
        },
    }


@pytest.fixture
def raw_email_simple():
    """Simple raw email for testing"""
    return b"""From: sender@example.com
To: recipient@example.com
Subject: Test Email
Date: Mon, 1 Jan 2024 12:00:00 +0000
Message-ID: <test@example.com>

This is a test email body.
"""


@pytest.fixture
def raw_email_html():
    """HTML email for testing markdown conversion"""
    return b"""From: sender@example.com
To: recipient@example.com
Subject: HTML Test Email
Date: Mon, 1 Jan 2024 12:00:00 +0000
Message-ID: <test-html@example.com>
Content-Type: text/html; charset="utf-8"

<html>
<body>
<h1>Test Header</h1>
<p>This is <strong>bold text</strong> and <em>italic text</em>.</p>
<p>Here's a <a href="http://example.com">link</a>.</p>
</body>
</html>
"""


@pytest.fixture
def mock_imap_client():
    """Mock IMAP client like inbox-zero's test setup"""
    mock = MagicMock()
    mock.login = Mock(return_value=True)
    mock.select_folder = Mock(return_value={'EXISTS': 10})
    mock.search = Mock(return_value=[1, 2, 3])
    mock.fetch = Mock(return_value={
        1: {b'RFC822': b"test email 1", b'FLAGS': []},
        2: {b'RFC822': b"test email 2", b'FLAGS': []},
        3: {b'RFC822': b"test email 3", b'FLAGS': []},
    })
    mock.logout = Mock()
    mock.add_flags = Mock()
    mock.copy = Mock()
    mock.expunge = Mock()
    mock.list_folders = Mock(return_value=[
        ([], b'/', 'INBOX'),
        ([], b'/', 'Sent'),
        ([], b'/', 'Trash'),
        ([], b'/', 'Archive'),
    ])
    
    return mock


@pytest.fixture
def mock_llm():
    """Mock LLM following inbox-zero patterns"""
    return AsyncMock(
        ainvoke=AsyncMock(
            return_value=ClassificationResult(
                category=EmailCategory.WORK,
                sender_type=SenderType.KNOWN,
                reply_status=ReplyStatus.NO_ACTION,
                is_urgent=False,
                confidence=0.95,
                reasoning="Test classification",
                action=EmailAction(type="keep")
            )
        )
    )


@pytest.fixture
def processed_email_sample():
    """Sample processed email"""
    return ProcessedEmail(
        uid="123",
        message_id="<test@example.com>",
        subject="Test Email",
        from_address="sender@example.com",
        to_addresses=["recipient@example.com"],
        date=datetime.now(),
        body_markdown="Test email body",
        attachment_texts=[],
        attachment_info=[],
        sender_domain="example.com",
        has_attachments=False,
        attachment_count=0,
        raw_headers={},
        folder="INBOX",
        flags=[],
    )

