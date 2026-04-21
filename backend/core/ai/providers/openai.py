"""
OpenAI Provider Implementation

Wraps LangChain's ChatOpenAI for backward compatibility.
"""

from .base import BaseLLMProvider, LLMResponse, TokenUsage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
import logging

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider using LangChain with structured outputs."""
    
    def __init__(self, model: str, temperature: float = 0.1):
        super().__init__(model, temperature)
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # GPT-5 series only supports temperature=1.0
        if model.startswith("gpt-5") and temperature != 1.0:
            logger.warning(f"{model} only supports temperature=1.0, overriding from {temperature}")
            temperature = 1.0
            self.temperature = 1.0
        
        # Initialize LangChain ChatOpenAI
        self.client = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key
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
        """Get structured completion from OpenAI via LangChain."""
        try:
            # Use LangChain's with_structured_output for Pydantic models
            # Enable usage tracking via callbacks
            structured_llm = self.client.with_structured_output(response_format)
            
            # Create prompt template
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are an expert email classifier."),
                ("user", "{input}")
            ])
            
            # Create chain
            chain = prompt_template | structured_llm
            
            # Invoke chain with callbacks to capture usage
            result = await chain.ainvoke({"input": prompt})
            
            # Extract token usage from LangChain response
            # Try multiple methods to get actual usage
            prompt_tokens = None
            completion_tokens = None
            
            # Method 1: Check response metadata
            try:
                if hasattr(result, 'response_metadata'):
                    metadata = result.response_metadata
                    if 'token_usage' in metadata:
                        prompt_tokens = metadata['token_usage'].get('prompt_tokens')
                        completion_tokens = metadata['token_usage'].get('completion_tokens')
            except:
                pass
            
            # Method 2: Check usage_metadata attribute
            if prompt_tokens is None:
                try:
                    usage_metadata = getattr(result, 'usage_metadata', None)
                    if usage_metadata:
                        prompt_tokens = usage_metadata.get('input_tokens', 0)
                        completion_tokens = usage_metadata.get('output_tokens', 0)
                except:
                    pass
            
            # Method 3: Fallback to improved estimation
            if prompt_tokens is None or completion_tokens is None:
                logger.warning(
                    f"Could not extract exact token usage from LangChain, using estimation"
                )
                # Improved estimation: count more accurately
                # System message + user prompt
                from ..config_constants import CHARS_PER_TOKEN_CONSERVATIVE
                estimated_prompt = len("You are an expert email classifier.") + len(prompt)
                prompt_tokens = estimated_prompt // CHARS_PER_TOKEN_CONSERVATIVE
                completion_tokens = len(str(result)) // CHARS_PER_TOKEN_CONSERVATIVE
            
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
            logger.error(f"OpenAI API error: {e}")
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

