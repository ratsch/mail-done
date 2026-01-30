"""
Reply Template System

Pre-written templates for common email response scenarios.
Provides 2-3 variations per category with variable substitution.
"""
from typing import Dict, List, Optional
from string import Template


class ReplyTemplates:
    """Template-based reply generation for common scenarios."""
    
    # Template structure: category-decision
    TEMPLATES = {
        # ============================================================
        # PhD APPLICATION REPLIES
        # ============================================================
        "application-phd-accept": {
            "subject_template": "Re: $original_subject",
            "variants": {
                "enthusiastic": Template("""Dear $applicant_name,

Thank you for your application and interest in joining our lab.

I'm very interested in your background in $research_area. Your work on $specific_detail aligns well with our current research directions.

I'd like to schedule a call to discuss potential PhD opportunities. Are you available for a 30-minute video call next week?

Please also send me:
- Your CV
- A brief research statement (1-2 pages)
- Your availability

Best regards,
$signature_name"""),
                
                "cautious": Template("""Dear $applicant_name,

Thank you for reaching out about PhD opportunities in our lab.

Your background in $research_area is interesting. Before we proceed, I'd like to learn more about your research interests and background.

Could you please send me:
1. Your current CV
2. A brief research statement explaining your interests (1 page)
3. Your availability for a short introductory call

Best,
$signature_name"""),
                
                "request_more_info": Template("""Dear $applicant_name,

Thank you for your interest in our lab.

To better assess the fit, could you please provide:
- Detailed CV with publications
- Research statement outlining your interests and how they align with our work
- Names of 2-3 references
- Timeline for starting PhD

I'll review these and get back to you.

Best regards,
$signature_name"""),
            }
        },
        
        "application-phd-decline": {
            "subject_template": "Re: $original_subject",
            "variants": {
                "polite": Template("""Dear $applicant_name,

Thank you for your interest in joining our lab.

Unfortunately, I don't currently have capacity for new PhD students. I encourage you to explore other opportunities - you might find a good fit in related groups.

Best of luck with your search,
$signature_name"""),
                
                "redirect": Template("""Dear $applicant_name,

Thank you for your application. While your background in $research_area is strong, it doesn't closely align with our current research directions.

I'd recommend reaching out to $other_professor who works more directly in $redirect_area. Their lab might be a better fit for your interests.

Best regards,
$signature_name"""),
                
                "timing": Template("""Dear $applicant_name,

Thank you for your interest in our lab.

While your background is interesting, I'm not planning to take on new PhD students in the near term. I encourage you to check back in $timeframe or explore other opportunities.

Best,
$signature_name"""),
            }
        },
        
        # ============================================================
        # POSTDOC APPLICATION REPLIES
        # ============================================================
        "application-postdoc-accept": {
            "subject_template": "Re: $original_subject",
            "variants": {
                "enthusiastic": Template("""Dear $applicant_name,

Thank you for your application for a postdoc position.

Your research on $research_topic is very relevant to our work. I'm particularly interested in your expertise in $specific_skill.

I'd like to discuss this further. Are you available for a video call next week to discuss:
- Your research interests and goals
- Potential projects in our lab
- Funding opportunities
- Timeline

Please send your availability.

Best regards,
$signature_name"""),
                
                "funding_dependent": Template("""Dear $applicant_name,

Thank you for your interest in a postdoc position in our lab.

Your background in $research_area is impressive. I'm currently exploring funding opportunities and would like to keep you in mind.

Could you send me:
- Your publication list
- Brief research proposal (2-3 pages)
- Your timeline and funding situation
- Whether you're applying for fellowships

I'll update you on funding status soon.

Best,
$signature_name"""),
            }
        },
        
        "application-postdoc-decline": {
            "subject_template": "Re: $original_subject",
            "variants": {
                "no_positions": Template("""Dear $applicant_name,

Thank you for your interest in a postdoc position.

Unfortunately, I don't have open postdoc positions at this time. I encourage you to check our lab website periodically for future opportunities.

Best of luck with your search,
$signature_name"""),
            }
        },
        
        # ============================================================
        # SPEAKING INVITATION REPLIES
        # ============================================================
        "invitation-speaking-accept": {
            "subject_template": "Re: $original_subject",
            "variants": {
                "confirm": Template("""Dear $organizer_name,

Thank you for the invitation to speak at $event_name on $event_date.

I'm happy to accept. I'll prepare a talk on $topic for the $duration slot.

Please send me:
- Any specific requirements for the talk
- Expected audience background
- Technical setup details
- Travel/accommodation information

Looking forward to it.

Best regards,
$signature_name"""),
                
                "tentative": Template("""Dear $organizer_name,

Thank you for the invitation to speak at $event_name.

I'm very interested, but need to check my schedule. Could you confirm:
- Exact date and time
- Duration of talk
- Whether virtual participation is possible
- Deadline for confirmation

I'll get back to you by $deadline.

Best,
$signature_name"""),
            }
        },
        
        "invitation-speaking-decline": {
            "subject_template": "Re: $original_subject",
            "variants": {
                "schedule_conflict": Template("""Dear $organizer_name,

Thank you for the invitation to speak at $event_name.

Unfortunately, I have a scheduling conflict on $event_date and won't be able to attend. 

I appreciate you thinking of me and hope we can find another opportunity to collaborate.

Best regards,
$signature_name"""),
                
                "capacity": Template("""Dear $organizer_name,

Thank you for the invitation to speak at $event_name.

While I'm honored by the invitation, I'm overcommitted with speaking engagements currently. I need to respectfully decline.

I'd be happy to suggest colleagues who might be a good fit: $suggestions.

Best,
$signature_name"""),
            }
        },
        
        # ============================================================
        # COMMITTEE INVITATION REPLIES
        # ============================================================
        "invitation-committee-accept": {
            "subject_template": "Re: $original_subject",
            "variants": {
                "confirm": Template("""Dear $organizer_name,

Thank you for the invitation to serve on $committee_name.

I'm happy to accept. Please send me:
- Committee charge and responsibilities
- Expected time commitment
- Meeting schedule
- Any preparation materials

I look forward to contributing.

Best regards,
$signature_name"""),
            }
        },
        
        "invitation-committee-decline": {
            "subject_template": "Re: $original_subject",
            "variants": {
                "capacity": Template("""Dear $organizer_name,

Thank you for the invitation to serve on $committee_name.

While I'm honored, I'm currently overcommitted with service obligations and need to respectfully decline.

I'd be happy to suggest colleagues who might be interested: $suggestions.

Best,
$signature_name"""),
            }
        },
        
        # ============================================================
        # REVIEW REQUEST REPLIES
        # ============================================================
        "review-peer-journal-accept": {
            "subject_template": "Re: $original_subject",
            "variants": {
                "confirm": Template("""Dear $editor_name,

Thank you for the invitation to review manuscript $manuscript_id for $journal_name.

I'm happy to accept. I'll complete the review by $deadline.

Please send the manuscript and any specific review guidelines.

Best regards,
$signature_name"""),
                
                "timeline_negotiate": Template("""Dear $editor_name,

Thank you for the invitation to review manuscript $manuscript_id.

I'm interested in reviewing this paper, but the deadline of $original_deadline is tight. Would it be possible to extend to $requested_deadline?

Please let me know if that works.

Best,
$signature_name"""),
            }
        },
        
        "review-peer-journal-decline": {
            "subject_template": "Re: $original_subject",
            "variants": {
                "conflict": Template("""Dear $editor_name,

Thank you for the invitation to review manuscript $manuscript_id.

Unfortunately, I have a conflict of interest with this work and should not review it. I can suggest $alternative_reviewers who might be suitable.

Best regards,
$signature_name"""),
                
                "capacity": Template("""Dear $editor_name,

Thank you for considering me to review manuscript $manuscript_id for $journal_name.

Unfortunately, I'm currently overcommitted with reviews and cannot take on additional ones at this time.

I can suggest alternative reviewers: $alternative_reviewers.

Best,
$signature_name"""),
                
                "not_expert": Template("""Dear $editor_name,

Thank you for the invitation to review manuscript $manuscript_id.

After reading the abstract, I don't think I'm the best reviewer for this work as it's outside my area of expertise. I suggest contacting $alternative_reviewers who work more directly in this area.

Best regards,
$signature_name"""),
            }
        },
        
        # ============================================================
        # GRANT REVIEW INVITATION REPLIES
        # ============================================================
        "review-grant-accept": {
            "subject_template": "Re: $original_subject",
            "variants": {
                "confirm": Template("""Dear $contact_name,

Thank you for the invitation to review proposals for $grant_program.

I'm happy to accept. Please send me:
- Review guidelines and criteria
- Timeline and deadlines
- Number of proposals to review
- Confidentiality agreement (if needed)

Best regards,
$signature_name"""),
            }
        },
        
        "review-grant-decline": {
            "subject_template": "Re: $original_subject",
            "variants": {
                "capacity": Template("""Dear $contact_name,

Thank you for the invitation to review proposals for $grant_program.

While I'm honored, I'm currently overcommitted and need to respectfully decline.

Best,
$signature_name"""),
            }
        },
        
        # ============================================================
        # EDITORIAL INVITATION REPLIES
        # ============================================================
        "invitation-editorial-accept": {
            "subject_template": "Re: $original_subject",
            "variants": {
                "confirm": Template("""Dear $editor_name,

Thank you for the invitation to serve on the editorial board of $journal_name.

I'm honored and happy to accept. Please send me:
- Responsibilities and expectations
- Time commitment details
- Term duration
- Next steps

I look forward to contributing.

Best regards,
$signature_name"""),
                
                "clarify": Template("""Dear $editor_name,

Thank you for the invitation to join the editorial board of $journal_name.

I'm very interested, but would like to clarify:
- Expected time commitment per year
- Handling load (manuscripts per year)
- Term length
- Any associate editor responsibilities

Once I have these details, I'll be able to confirm.

Best,
$signature_name"""),
            }
        },
        
        "invitation-editorial-decline": {
            "subject_template": "Re: $original_subject",
            "variants": {
                "capacity": Template("""Dear $editor_name,

Thank you for the invitation to serve on the editorial board of $journal_name.

I'm honored, but I'm currently on several editorial boards and need to limit additional commitments. I must respectfully decline.

Best regards,
$signature_name"""),
            }
        },
        
        # ============================================================
        # WORK / COLLEAGUE EMAILS
        # ============================================================
        "work-colleague-acknowledge": {
            "subject_template": "Re: $original_subject",
            "variants": {
                "confirm": Template("""Dear $colleague_name,

Thanks for reaching out about $topic.

$response

Best,
$signature_name"""),
                
                "schedule_meeting": Template("""Dear $colleague_name,

Thanks for your email about $topic.

I'd be happy to discuss this. Are you available for a meeting $timeframe? Please send some time slots that work for you.

Best,
$signature_name"""),
            }
        },
        
        "work-student-acknowledge": {
            "subject_template": "Re: $original_subject",
            "variants": {
                "confirm": Template("""Hi $student_name,

Thanks for your email.

$response

Let me know if you have questions.

Best,
$signature_name"""),
            }
        },
        
        # ============================================================
        # GENERAL / ACKNOWLEDGMENT
        # ============================================================
        "general-acknowledge": {
            "subject_template": "Re: $original_subject",
            "variants": {
                "received": Template("""Dear $sender_name,

Thank you for your email. I've received it and will get back to you soon.

Best regards,
$signature_name"""),
                
                "need_time": Template("""Dear $sender_name,

Thank you for your email about $topic.

I need some time to review this properly. I'll get back to you by $timeline.

Best,
$signature_name"""),
            }
        },
    }
    
    def get_template(
        self,
        category: str,
        decision: str,
        tone: str = "confirm"
    ) -> Optional[Template]:
        """
        Get a specific template.
        
        Args:
            category: Email category (e.g., "application-phd")
            decision: Response decision (accept/decline/acknowledge)
            tone: Template variant (enthusiastic/polite/cautious/etc.)
        
        Returns:
            Template object or None if not found
        """
        template_key = f"{category}-{decision}"
        
        if template_key not in self.TEMPLATES:
            return None
        
        template_data = self.TEMPLATES[template_key]
        
        # Get variant
        if tone in template_data["variants"]:
            return template_data["variants"][tone]
        
        # Fall back to first variant
        return list(template_data["variants"].values())[0]
    
    def get_subject(
        self,
        category: str,
        decision: str,
        original_subject: str
    ) -> str:
        """Get subject line for reply."""
        template_key = f"{category}-{decision}"
        
        if template_key not in self.TEMPLATES:
            return f"Re: {original_subject}"
        
        subject_template = self.TEMPLATES[template_key].get("subject_template", "Re: $original_subject")
        return Template(subject_template).safe_substitute(original_subject=original_subject)
    
    def get_available_tones(
        self,
        category: str,
        decision: str
    ) -> List[str]:
        """Get available tone variants for a template."""
        template_key = f"{category}-{decision}"
        
        if template_key not in self.TEMPLATES:
            return []
        
        return list(self.TEMPLATES[template_key]["variants"].keys())
    
    def generate_from_template(
        self,
        category: str,
        decision: str,
        tone: str,
        context: Dict[str, str]
    ) -> Optional[str]:
        """
        Generate reply from template with variable substitution.
        
        Args:
            category: Email category
            decision: Response decision
            tone: Template variant
            context: Dictionary of variables for substitution
        
        Returns:
            Generated email body or None
        """
        template = self.get_template(category, decision, tone)
        
        if not template:
            return None
        
        try:
            return template.safe_substitute(**context)
        except Exception as e:
            # If substitution fails, return template with available vars
            return template.safe_substitute(**context)
    
    def list_available_templates(self) -> Dict[str, List[str]]:
        """List all available templates."""
        templates_by_category = {}
        
        for template_key in self.TEMPLATES.keys():
            category_decision = template_key
            if category_decision not in templates_by_category:
                templates_by_category[category_decision] = []
            
            templates_by_category[category_decision] = list(
                self.TEMPLATES[template_key]["variants"].keys()
            )
        
        return templates_by_category

