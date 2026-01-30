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
        
        # Pricing (per 1M tokens, updated Nov 2025)
        self.pricing = {
            # GPT-4 series
            "gpt-4o-mini": {"input": 0.150, "output": 0.600},
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
            
            # GPT-4.1 series (better prompt caching: 75% off cached tokens)
            "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
            "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
            "gpt-4.1": {"input": 2.00, "output": 8.00},
            
            # GPT-5 series (released August 2025)
            "gpt-5-nano": {"input": 0.05, "output": 0.40},
            "gpt-5-mini": {"input": 0.25, "output": 2.00},
            "gpt-5": {"input": 1.25, "output": 10.00},
            "gpt-5.1": {"input": 1.25, "output": 10.00},  # GPT-5.1 (Nov 2025)
            "gpt-5.1-instant": {"input": 1.00, "output": 8.00},
            "gpt-5.1-thinking": {"input": 1.50, "output": 12.00},
            
            # o1 series (reasoning models)
            "o1-preview": {"input": 15.00, "output": 60.00},
            "o1-mini": {"input": 3.00, "output": 12.00},
        }
        
        if model not in self.pricing:
            logger.warning(f"Unknown model {model}, using gpt-4o-mini pricing")
    
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
        pricing = self.pricing.get(self.model, self.pricing["gpt-4o-mini"])
        input_cost = (usage.prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (usage.completion_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

