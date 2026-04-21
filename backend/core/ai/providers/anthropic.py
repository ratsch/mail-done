"""
Anthropic Provider Implementation

Wraps LangChain's ChatAnthropic for Claude models.
"""

from .base import BaseLLMProvider, LLMResponse, TokenUsage
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
import os
import logging

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic/Claude provider using LangChain with structured outputs."""
    
    def __init__(self, model: str, temperature: float = 0.1):
        super().__init__(model, temperature)
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        # Initialize LangChain ChatAnthropic
        self.client = ChatAnthropic(
            model=model,
            temperature=temperature,
            anthropic_api_key=api_key
        )
        
        # Pricing pulled from the shared source of truth — never maintain
        # a second copy here; see backend/core/ai/pricing.py.
        from backend.core.ai.pricing import MODEL_PRICING
        self.pricing = MODEL_PRICING

        if model not in self.pricing:
            logger.warning(f"Unknown model {model}, using fallback pricing")
    
    async def _complete_impl(
        self, 
        prompt: str, 
        response_format,
        **kwargs
    ) -> LLMResponse:
        """Get structured completion from Anthropic via LangChain."""
        try:
            # Use LangChain's with_structured_output for Pydantic models
            structured_llm = self.client.with_structured_output(response_format)
            
            # Create prompt template
            prompt_template = ChatPromptTemplate.from_messages([
                ("user", "{input}")
            ])
            
            # Create chain
            chain = prompt_template | structured_llm
            
            # Invoke chain
            result = await chain.ainvoke({"input": prompt})
            
            # Extract token usage from LangChain response
            try:
                # Try to get actual usage from the response metadata
                usage_metadata = getattr(result, '_usage_metadata', None)
                if usage_metadata:
                    prompt_tokens = usage_metadata.get('input_tokens', 0)
                    completion_tokens = usage_metadata.get('output_tokens', 0)
                else:
                    # Fallback: estimate based on string lengths
                    prompt_tokens = len(prompt) // 4  # Rough estimate
                    completion_tokens = len(str(result)) // 4
            except:
                # If all else fails, use rough estimates
                prompt_tokens = len(prompt) // 4
                completion_tokens = len(str(result)) // 4
            
            usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
            
            return LLMResponse(
                parsed=result,
                usage=usage,
                raw_response={"result": str(result)},
                latency_ms=0  # Set by base class
            )
        
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    def calculate_cost(self, usage: TokenUsage) -> float:
        """Calculate cost based on token usage."""
        from backend.core.ai.pricing import compute_cost
        return compute_cost(
            model_name=self.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cached_tokens=getattr(usage, "cached_tokens", 0) or 0,
        )

