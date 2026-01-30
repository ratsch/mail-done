"""
AI-Powered Reply Generator

Uses LLM to generate personalized email responses based on:
- Email content and context
- AI classification metadata
- Sender history
- Your writing style
"""
import os
import re
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from backend.core.database.models import Email, EmailMetadata, SenderHistory
from backend.core.prompt_loader import get_prompt

logger = logging.getLogger(__name__)


class ReplyDraft(BaseModel):
    """Generated reply draft"""
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body text")
    tone: str = Field(description="Tone used (professional/friendly/formal)")
    confidence: float = Field(description="Confidence in quality (0-1)")
    reasoning: str = Field(description="Why this approach was chosen")


class EmailContext(BaseModel):
    """Extracted context from email for personalization"""
    sender_name: str = Field(description="Sender's name")
    sender_institution: Optional[str] = Field(default=None, description="Sender's institution/company")
    
    # For applications
    applicant_background: Optional[str] = Field(default=None, description="Brief background summary")
    research_area: Optional[str] = Field(default=None, description="Research area/interests")
    specific_detail: Optional[str] = Field(default=None, description="Notable specific detail to mention")
    
    # For invitations
    event_name: Optional[str] = Field(default=None, description="Event/conference name")
    event_date: Optional[str] = Field(default=None, description="Event date")
    event_location: Optional[str] = Field(default=None, description="Event location")
    topic: Optional[str] = Field(default=None, description="Topic/role requested")
    duration: Optional[str] = Field(default=None, description="Talk duration")
    
    # For reviews
    journal_name: Optional[str] = Field(default=None, description="Journal name")
    manuscript_id: Optional[str] = Field(default=None, description="Manuscript ID")
    deadline: Optional[str] = Field(default=None, description="Review deadline")
    
    # General
    organizer_name: Optional[str] = Field(default=None, description="Organizer/editor name")
    committee_name: Optional[str] = Field(default=None, description="Committee name")


class AIReplyGenerator:
    """LLM-based reply generation with context awareness."""

    # Default fallback prompt (loaded from config/prompts.yaml if available)
    _DEFAULT_SYSTEM_PROMPT = """You are helping an academic researcher compose professional email replies.

The researcher leads a research lab in a technical/scientific field. They receive many emails:
- PhD/Postdoc applications
- Speaking and committee invitations
- Review requests (papers, grants)
- Colleague inquiries

Writing Style Guidelines:
- Professional but friendly tone
- Concise (2-3 paragraphs maximum)
- Direct and clear
- For applications: acknowledge specific details from their email
- For invitations: be gracious whether accepting or declining
- For reviews: set clear expectations on timeline
- Sign off with "Best regards," or "Best," depending on formality

Key Principles:
1. Personalize by referencing specific details from the email
2. Be respectful of applicant's/sender's time
3. Set clear expectations (next steps, timeline)
4. Maintain professional boundaries
5. Be encouraging when declining (suggest alternatives if appropriate)

Generate concise, professional replies that sound natural and human."""

    @property
    def SYSTEM_PROMPT(self) -> str:
        """Get system prompt from config or use default."""
        return get_prompt("reply_generator.system_prompt", default=self._DEFAULT_SYSTEM_PROMPT)

    def __init__(self, model: str = "gpt-4o-mini", provider: str = "openai"):
        """
        Initialize AI reply generator.
        
        Args:
            model: Model to use (gpt-4o-mini, gpt-4o, claude-3-haiku, etc.)
            provider: LLM provider (openai or anthropic)
        """
        self.model = model
        self.provider = provider
        
        # Initialize LLM
        if provider == "openai":
            self.llm = ChatOpenAI(
                model=model,
                temperature=0.7,  # Some creativity, but consistent
                api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
            self.llm = ChatAnthropic(
                model=model,
                temperature=0.7,
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
    
    async def generate_reply(
        self,
        email: Email,
        metadata: EmailMetadata,
        decision: str,  # accept/decline/maybe/acknowledge
        tone: str = "professional",
        sender_history: Optional[SenderHistory] = None,
        num_variations: int = 2
    ) -> List[ReplyDraft]:
        """
        Generate personalized reply drafts using LLM.
        
        Args:
            email: Email to reply to
            metadata: AI classification metadata
            decision: Response decision (accept/decline/maybe/acknowledge)
            tone: Desired tone (professional/friendly/formal)
            sender_history: Sender's history (optional)
            num_variations: Number of variations to generate
        
        Returns:
            List of ReplyDraft objects
        """
        # Extract context from email
        context = self._extract_context(email, metadata)
        
        # Build category-specific prompt
        category = metadata.ai_category or "general"
        prompt_text = self._build_reply_prompt(
            email=email,
            metadata=metadata,
            context=context,
            decision=decision,
            tone=tone,
            category=category,
            sender_history=sender_history
        )
        
        # Generate variations
        drafts = []
        for i in range(num_variations):
            # Vary temperature slightly for different options
            temp_llm = self.llm
            if i > 0:
                temp = 0.7 + (i * 0.1)  # 0.7, 0.8, 0.9
                if self.provider == "openai":
                    temp_llm = ChatOpenAI(model=self.model, temperature=temp)
                else:
                    temp_llm = ChatAnthropic(model=self.model, temperature=temp)
            
            # Generate draft
            draft = await self._generate_single_draft(
                prompt_text=prompt_text,
                llm=temp_llm,
                tone=tone,
                variation_num=i+1
            )
            
            if draft:
                drafts.append(draft)
        
        return drafts
    
    def _extract_context(
        self,
        email: Email,
        metadata: EmailMetadata
    ) -> EmailContext:
        """
        Extract key details from email for personalization.
        
        Args:
            email: Email object
            metadata: Email metadata
        
        Returns:
            EmailContext with extracted details
        """
        body = email.body_markdown or email.body_text or ""
        
        # Truncate body for extraction if extremely long (safety check)
        # We only need enough context to extract key details, not the entire email
        MAX_EXTRACTION_CHARS = 50000  # ~12k tokens, more than enough for extraction
        if len(body) > MAX_EXTRACTION_CHARS:
            body = body[:MAX_EXTRACTION_CHARS]
            logger.debug(f"Truncated email body for context extraction: {len(body):,} chars")
        
        subject = email.subject
        
        # Extract sender name (from "Name <email>" format)
        sender_name = email.from_name or email.from_address.split('@')[0]
        
        # Extract institution/affiliation
        institution = self._extract_institution(email.from_address, body)
        
        # Category-specific extraction
        category = metadata.ai_category or ""
        
        context = EmailContext(
            sender_name=sender_name,
            sender_institution=institution
        )
        
        # Application-specific
        if category.startswith('application'):
            context.research_area = self._extract_research_area(body, subject)
            context.specific_detail = self._extract_specific_detail(body, category)
            context.applicant_background = self._extract_background(body)
        
        # Invitation-specific
        elif category.startswith('invitation'):
            context.event_name = self._extract_event_name(subject, body)
            context.event_date = self._extract_event_date(body)
            context.event_location = self._extract_location(body)
            context.topic = self._extract_topic(body, subject)
            context.organizer_name = sender_name
        
        # Review-specific
        elif category.startswith('review'):
            context.journal_name = self._extract_journal(subject, body)
            context.manuscript_id = self._extract_manuscript_id(subject, body)
            context.deadline = self._extract_deadline(body)
            context.organizer_name = sender_name
        
        return context
    
    def _extract_institution(self, email_address: str, body: str) -> Optional[str]:
        """Extract institution from email or body."""
        # Common university domains
        domain = email_address.split('@')[-1] if '@' in email_address else ""
        
        # Map domains to institutions
        institution_map = {
            'stanford.edu': 'Stanford University',
            'mit.edu': 'MIT',
            'harvard.edu': 'Harvard University',
            'berkeley.edu': 'UC Berkeley',
            # Add your institution domain mappings here
            # 'example.edu': 'Example University',
            'uzh.ch': 'University of Zurich',
            'cam.ac.uk': 'Cambridge University',
            'ox.ac.uk': 'Oxford University',
        }
        
        for key, inst in institution_map.items():
            if key in domain:
                return inst
        
        # Try to find in signature
        patterns = [
            r'(?:PhD student|Postdoc|Professor) (?:at|@) ([A-Z][A-Za-z\s]+University)',
            r'([A-Z][A-Za-z\s]+University)',
            r'([A-Z][A-Za-z\s]+Institute)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, body)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_research_area(self, body: str, subject: str) -> Optional[str]:
        """Extract research area from application."""
        # Look for common patterns
        patterns = [
            r'research (?:interest|area|focus)(?:s)? (?:in|on|include) ([^.]+)',
            r'working on ([^.]+)',
            r'interested in ([^.]+)',
            r'PhD in ([^.]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, body.lower())
            if match:
                area = match.group(1).strip()
                if len(area) < 100:  # Reasonable length
                    return area
        
        # Check subject for keywords
        ml_keywords = ['machine learning', 'deep learning', 'AI', 'neural networks']
        bio_keywords = ['biology', 'genomics', 'bioinformatics', 'computational biology']
        
        for keyword in ml_keywords + bio_keywords:
            if keyword.lower() in (subject + body).lower():
                return keyword
        
        return "computational biology"  # Default
    
    def _extract_specific_detail(self, body: str, category: str) -> Optional[str]:
        """Extract a specific detail to mention in reply."""
        # Look for publications
        pub_pattern = r'published (?:in |on )?([^.]+)'
        match = re.search(pub_pattern, body.lower())
        if match:
            return f"your work on {match.group(1).strip()}"
        
        # Look for projects
        project_pattern = r'worked on ([^.]+project[^.]*)'
        match = re.search(project_pattern, body.lower())
        if match:
            return match.group(1).strip()
        
        # Look for skills
        skill_pattern = r'experience (?:in|with) ([^.]+)'
        match = re.search(skill_pattern, body.lower())
        if match:
            return f"your experience in {match.group(1).strip()}"
        
        return None
    
    def _extract_background(self, body: str) -> Optional[str]:
        """Extract brief background summary."""
        # Look for current position
        position_patterns = [
            r'I am (?:a |an |currently )?([^.]+?(?:student|postdoc|researcher)[^.]*)',
            r'currently (?:a )?([^.]+?(?:at|in)[^.]*)',
        ]
        
        for pattern in position_patterns:
            match = re.search(pattern, body, re.IGNORECASE)
            if match:
                bg = match.group(1).strip()
                if len(bg) < 150:
                    return bg
        
        return None
    
    def _extract_event_name(self, subject: str, body: str) -> Optional[str]:
        """Extract event/conference name."""
        # Common patterns in invitations
        patterns = [
            r'(?:speak at|attend|invited to|participate in) ([A-Z][A-Za-z\s]+(?:Conference|Workshop|Symposium|Summit))',
            r'([A-Z][A-Z0-9]+\s+\d{4})',  # Acronyms like "ICML 2024"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, subject + " " + body)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_event_date(self, body: str) -> Optional[str]:
        """Extract event date."""
        # Date patterns
        patterns = [
            r'(?:on|scheduled for) ([A-Z][a-z]+ \d{1,2}(?:st|nd|rd|th)?,? \d{4})',
            r'(\d{1,2}/\d{1,2}/\d{4})',
            r'([A-Z][a-z]+ \d{4})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, body)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_location(self, body: str) -> Optional[str]:
        """Extract event location."""
        patterns = [
            r'(?:in|at|held in) ([A-Z][a-z]+(?:, [A-Z][a-z]+)?)',
            r'location: ([^.\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, body)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_topic(self, body: str, subject: str) -> Optional[str]:
        """Extract talk topic."""
        patterns = [
            r'talk (?:on|about) ([^.]+)',
            r'present (?:on|about) ([^.]+)',
            r'speak about ([^.]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, body.lower())
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_journal(self, subject: str, body: str) -> Optional[str]:
        """Extract journal name."""
        patterns = [
            r'(?:for|in) ([A-Z][A-Za-z\s]+(?:Journal|Review|Letters))',
            r'manuscript (?:for|to) ([A-Z][A-Za-z\s]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, subject + " " + body)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_manuscript_id(self, subject: str, body: str) -> Optional[str]:
        """Extract manuscript ID."""
        patterns = [
            r'manuscript (?:number|ID|#):? ([A-Z0-9-]+)',
            r'MS[- ]?([A-Z0-9-]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, subject + " " + body)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_deadline(self, body: str) -> Optional[str]:
        """Extract deadline."""
        patterns = [
            r'deadline[:\s]+([A-Z][a-z]+ \d{1,2}(?:st|nd|rd|th)?,? \d{4})',
            r'(?:by|before) ([A-Z][a-z]+ \d{1,2}(?:st|nd|rd|th)?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, body)
            if match:
                return match.group(1)
        
        return None
    
    def _build_reply_prompt(
        self,
        email: Email,
        metadata: EmailMetadata,
        context: EmailContext,
        decision: str,
        tone: str,
        category: str,
        sender_history: Optional['SenderHistory'] = None
    ) -> str:
        """Build LLM prompt for reply generation."""
        
        # Build sender context if available
        sender_context = ""
        if sender_history:
            if sender_history.email_count > 1:
                sender_context = f"\n\nSender History:\n"
                sender_context += f"- Previous emails: {sender_history.email_count}\n"
                if sender_history.avg_reply_time_hours:
                    sender_context += f"- Your typical response time: {sender_history.avg_reply_time_hours:.1f} hours\n"
                if sender_history.sender_type:
                    sender_context += f"- Relationship: {sender_history.sender_type}\n"
                if sender_history.always_replies:
                    sender_context += "- You typically respond to this sender\n"
                sender_context += "\nNote: This is not a first-time contact. You can reference previous interactions if appropriate."
        
        # Category-specific guidance
        category_guidance = {
            'application-phd': """This is a PhD application. 
If accepting: Show genuine interest, mention specific aspects of their background, set clear next steps.
If declining: Be respectful and encouraging, suggest alternatives if appropriate.""",
            
            'application-postdoc': """This is a postdoc application.
If accepting: Indicate interest, discuss funding/timeline, request additional materials.
If declining: Be professional, mention capacity or fit issues.""",
            
            'invitation-speaking': """This is a speaking invitation.
If accepting: Confirm interest, ask for logistics (audience, duration, technical setup).
If declining: Be gracious, mention schedule conflicts or capacity.""",
            
            'invitation-committee': """This is a committee/board invitation.
If accepting: Confirm and ask for details on time commitment, responsibilities.
If declining: Cite capacity issues, suggest alternatives if appropriate.""",
            
            'review-peer-journal': """This is a paper review request.
If accepting: Confirm timeline, ask for manuscript and guidelines.
If declining: Mention conflicts, capacity, or expertise gaps. Suggest alternative reviewers.""",
            
            'review-grant': """This is a grant review invitation.
If accepting: Confirm and request details on process and timeline.
If declining: Cite capacity constraints.""",
        }
        
        guidance = category_guidance.get(category, "This is a professional email requiring a response.")
        
        # Build context string
        context_str = f"Sender: {context.sender_name}"
        if context.sender_institution:
            context_str += f" ({context.sender_institution})"
        
        if context.research_area:
            context_str += f"\nResearch area: {context.research_area}"
        if context.specific_detail:
            context_str += f"\nNotable detail: {context.specific_detail}"
        if context.event_name:
            context_str += f"\nEvent: {context.event_name}"
        if context.event_date:
            context_str += f"\nDate: {context.event_date}"
        
        prompt = f"""{self.SYSTEM_PROMPT}

{guidance}
{sender_context}

Original Email:
From: {email.from_address}
Subject: {email.subject}
Date: {email.date.strftime('%Y-%m-%d')}

{context_str}

Your task: Write a {tone} reply that {decision}s this email.

Requirements:
1. Reference specific details from their email (use the context above)
2. Be personable but professional
3. Keep it concise (2-3 short paragraphs)
4. Set clear next steps or close politely
5. Use "Best regards," or "Best," as sign-off

Generate only the email body (no subject line). Start directly with "Dear {context.sender_name}," and write naturally."""

        return prompt
    
    async def _generate_single_draft(
        self,
        prompt_text: str,
        llm: ChatOpenAI,
        tone: str,
        variation_num: int
    ) -> Optional[ReplyDraft]:
        """Generate a single draft using LLM."""
        try:
            # Create chat prompt
            messages = [
                ("system", self.SYSTEM_PROMPT),
                ("user", prompt_text)
            ]
            
            prompt = ChatPromptTemplate.from_messages(messages)
            
            # Generate
            chain = prompt | llm
            response = await chain.ainvoke({})
            
            body = response.content.strip()
            
            # Generate subject (simple Re: pattern)
            subject = "Re: [subject]"
            
            # Estimate confidence (simple heuristic)
            confidence = 0.85 if len(body) > 100 and len(body) < 500 else 0.7
            
            reasoning = f"Generated using {self.model}, variation {variation_num}, {tone} tone"
            
            return ReplyDraft(
                subject=subject,
                body=body,
                tone=tone,
                confidence=confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error generating draft: {e}", exc_info=True)
            return None

