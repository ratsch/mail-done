"""
Helper module for extracting curated sender names from emails using LLM.
"""
import os
import logging
from typing import Optional
from backend.core.email.models import ProcessedEmail

logger = logging.getLogger(__name__)


async def extract_curated_sender_name(processed_email: ProcessedEmail) -> Optional[str]:
    """
    Extract curated sender name from email using lightweight LLM call.
    
    Uses the same provider stack as embeddings (Azure OpenAI if configured, otherwise OpenAI).
    Model defaults to gpt-5-nano for cost efficiency.
    
    Args:
        processed_email: ProcessedEmail object with email content
        
    Returns:
        Curated sender name string or None if extraction fails
    """
    try:
        # Prefer Azure OpenAI if configured (same provider infra as embeddings)
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if azure_api_key and os.getenv("AZURE_OPENAI_ENDPOINT"):
            from openai import AzureOpenAI
            client = AzureOpenAI(
                api_key=azure_api_key,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )
            model = os.getenv("AZURE_OPENAI_NAME_EXTRACT_MODEL", "gpt-5-mini")
        else:
            # Fallback to OpenAI
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.debug("LLM name extraction skipped: no OpenAI credentials")
                return None
            client = OpenAI(api_key=api_key)
            model = os.getenv("OPENAI_NAME_EXTRACT_MODEL", "gpt-5-mini")
        
        # Build compact prompt with available info
        from_name = processed_email.from_name or ""
        from_addr = processed_email.from_address or ""
        subject = processed_email.subject or ""
        body = processed_email.body_markdown or ""
        body = body.strip()
        if len(body) > 1600:
            body = body[:1600]
        
        # Extract attachment filenames if available
        attachment_filenames = []
        if processed_email.attachment_info:
            for att_info in processed_email.attachment_info:
                filename = att_info.filename if hasattr(att_info, 'filename') else att_info.get('filename', '')
                if filename:
                    attachment_filenames.append(filename)
        
        user_prompt = (
            "Extract the sender's full personal name (first and last) from the following email information.\n"
            "- If you find multiple variants, pick the one most likely to be the sender's real name.\n"
            "- Use the email body (signature or introduction) over the address if possible.\n"
            "- Attachment filenames may contain the sender's name (e.g., 'CV_John_Smith.pdf').\n"
            "- Return ONLY the name text (no quotes, no extra text). If unknown, return Unknown.\n\n"
            f"From name: {from_name}\n"
            f"From address: {from_addr}\n"
            f"Subject: {subject}\n"
        )
        
        # Add attachment filenames if available
        if attachment_filenames:
            user_prompt += f"Attachment filenames: {', '.join(attachment_filenames)}\n"
        
        user_prompt += (
            "Body:\n"
            f"{body}\n"
        )
        
        messages = [
            {"role": "system", "content": "You extract concise facts. Return only the answer string."},
            {"role": "user", "content": user_prompt},
        ]
        
        # GPT-5/o1 models usually require temperature=1.0 and support reasoning_effort
        is_reasoning_model = "gpt-5" in model or "o1" in model or "o3" in model
        completion_kwargs = {
            "model": model,
            "messages": messages,
        }
        
        if is_reasoning_model:
            completion_kwargs["temperature"] = 1.0
            completion_kwargs["reasoning_effort"] = "minimal"
            # Do not set max_tokens or max_completion_tokens for GPT-5/reasoning models
        else:
            completion_kwargs["temperature"] = 0.0
            completion_kwargs["max_tokens"] = 32
        
        try:
            resp = client.chat.completions.create(**completion_kwargs)
            except Exception as e:
                # Check for DeploymentNotFound (404), Bad Request (400), or similar model errors
                error_str = str(e).lower()
                if any(x in error_str for x in ["404", "400", "deploymentnotfound", "model does not exist", "bad request"]):
                    logger.warning(f"Model '{model}' failed with error: {e}. Falling back to 'gpt-4o-mini'.")
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        temperature=0.0,
                        max_tokens=32,
                    )
                else:
                    raise e
                
        text = resp.choices[0].message.content.strip() if resp and resp.choices else ""
        # Basic sanitize
        if text and text.lower() != "unknown":
            # Remove any wrapping quotes
            text = text.strip('"\''" ")
            # Reject email-like outputs
            if '@' in text:
                return None
            # Must be at least two tokens ideally; allow single token if looks like a name (capitalized)
            tokens = text.split()
            if len(tokens) >= 2 or (len(tokens) == 1 and tokens[0][0:1].isupper() and len(tokens[0]) >= 3):
                return text
        return None
    except Exception as e:
        logger.debug(f"LLM sender name extraction failed: {e}")
        return None


