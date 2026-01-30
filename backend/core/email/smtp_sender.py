"""
SMTP Email Sender

Handles sending emails via SMTP, including draft replies.
Integrates with IMAP to save to Sent folder.
"""
import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr, make_msgid
from typing import Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class SMTPSender:
    """Send emails via SMTP and sync with IMAP Sent folder."""
    
    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_username: Optional[str] = None,
        smtp_password: Optional[str] = None,
        from_name: Optional[str] = None,
        from_email: Optional[str] = None,
        use_tls: bool = True
    ):
        """
        Initialize SMTP sender.
        
        Args:
            smtp_host: SMTP server hostname
            smtp_port: SMTP server port (587 for TLS, 465 for SSL)
            smtp_username: SMTP username (usually same as email)
            smtp_password: SMTP password or app-specific password
            from_name: Display name for From field
            from_email: Email address for From field
            use_tls: Use TLS encryption (recommended)
        """
        self.smtp_host = smtp_host or os.getenv('SMTP_HOST')
        self.smtp_port = smtp_port or int(os.getenv('SMTP_PORT', '587'))
        self.smtp_username = smtp_username or os.getenv('SMTP_USERNAME')
        self.smtp_password = smtp_password or os.getenv('SMTP_PASSWORD')
        self.from_name = from_name or os.getenv('FROM_NAME', '')
        self.from_email = from_email or os.getenv('FROM_EMAIL', self.smtp_username)
        self.use_tls = use_tls
        
        if not all([self.smtp_host, self.smtp_username, self.smtp_password]):
            logger.warning(
                "SMTP not fully configured. Set SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD "
                "environment variables to enable email sending."
            )
    
    def send_email(
        self,
        to_address: str,
        subject: str,
        body: str,
        cc_addresses: Optional[List[str]] = None,
        bcc_addresses: Optional[List[str]] = None,
        in_reply_to: Optional[str] = None,
        references: Optional[str] = None,
        is_html: bool = False
    ) -> Optional[str]:
        """
        Send an email via SMTP.
        
        Args:
            to_address: Recipient email address
            subject: Email subject
            body: Email body (plain text or HTML)
            cc_addresses: CC recipients
            bcc_addresses: BCC recipients
            in_reply_to: Message-ID this is replying to (for threading)
            references: References header (for threading)
            is_html: Whether body is HTML
        
        Returns:
            Message-ID of sent email, or None if failed
        """
        if not all([self.smtp_host, self.smtp_username, self.smtp_password]):
            logger.error("SMTP not configured. Cannot send email.")
            return None
        
        try:
            # Create message
            if is_html:
                msg = MIMEMultipart('alternative')
                msg.attach(MIMEText(body, 'plain'))
                msg.attach(MIMEText(body, 'html'))
            else:
                msg = MIMEText(body, 'plain', 'utf-8')
            
            # Set headers
            msg['From'] = formataddr((self.from_name, self.from_email))
            msg['To'] = to_address
            msg['Subject'] = subject
            
            # Generate unique Message-ID
            message_id = make_msgid(domain=self.from_email.split('@')[1])
            msg['Message-ID'] = message_id
            
            # CC/BCC
            if cc_addresses:
                msg['Cc'] = ', '.join(cc_addresses)
            if bcc_addresses:
                msg['Bcc'] = ', '.join(bcc_addresses)
            
            # Threading headers (for replies)
            if in_reply_to:
                msg['In-Reply-To'] = in_reply_to
            if references:
                msg['References'] = references
            elif in_reply_to:
                # If no references but replying, use in_reply_to
                msg['References'] = in_reply_to
            
            # Add date
            msg['Date'] = datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S +0000')
            
            # Build recipient list
            recipients = [to_address]
            if cc_addresses:
                recipients.extend(cc_addresses)
            if bcc_addresses:
                recipients.extend(bcc_addresses)
            
            # Connect to SMTP server
            if self.use_tls:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30)
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, timeout=30)
            
            # Login
            server.login(self.smtp_username, self.smtp_password)
            
            # Send
            server.send_message(msg, from_addr=self.from_email, to_addrs=recipients)
            server.quit()
            
            logger.info(f"Email sent successfully to {to_address}: {subject}")
            logger.debug(f"Message-ID: {message_id}")
            
            return message_id
            
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error sending email to {to_address}: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Error sending email to {to_address}: {e}", exc_info=True)
            return None
    
    def send_reply(
        self,
        original_message_id: str,
        original_subject: str,
        original_references: Optional[str],
        to_address: str,
        subject: str,
        body: str,
        cc_addresses: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Send a reply to an email (with proper threading headers).
        
        Args:
            original_message_id: Message-ID of email being replied to
            original_subject: Subject of original email
            original_references: References header from original
            to_address: Reply-to address
            subject: Reply subject (should be "Re: original")
            body: Reply body
            cc_addresses: CC recipients
        
        Returns:
            Message-ID of sent reply
        """
        # Build references chain
        if original_references:
            references = f"{original_references} {original_message_id}"
        else:
            references = original_message_id
        
        # Send with threading headers
        return self.send_email(
            to_address=to_address,
            subject=subject if subject.startswith('Re:') else f"Re: {original_subject}",
            body=body,
            cc_addresses=cc_addresses,
            in_reply_to=original_message_id,
            references=references,
            is_html=False
        )
