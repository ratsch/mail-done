"""
Base LLM Provider Interface

Defines the abstract interface that all LLM providers must implement.
"""

from abc import ABC, abstractmethod
from typing import Type, Optional
from pydantic import BaseModel
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Token usage information from LLM API call."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class LLMResponse:
    """Unified response from any LLM provider."""
    parsed: BaseModel  # Structured output (AIClassificationResult)
    usage: TokenUsage
    raw_response: dict  # Original response for debugging
    latency_ms: int  # Response time in milliseconds


class BaseLLMProvider(ABC):
    """
    Base interface for LLM providers.
    
    All providers (OpenAI, Anthropic, etc.) must implement this interface.
    """
    
    def __init__(self, model: str, temperature: float = 0.1):
        self.model = model
        self.temperature = temperature
        
        # Usage tracking
        self.total_requests = 0
        self.total_tokens = 0
        self.total_cost = 0.0
    
    async def complete(
        self, 
        prompt: str, 
        response_format: Type[BaseModel],
        **kwargs
    ) -> LLMResponse:
        """
        Get structured completion from LLM.
        
        Args:
            prompt: User prompt
            response_format: Pydantic model for structured output
            **kwargs: Provider-specific arguments
            
        Returns:
            LLMResponse with parsed result and usage
        """
        start_time = time.time()
        
        # Call provider-specific implementation
        response = await self._complete_impl(prompt, response_format, **kwargs)
        
        # Add timing
        response.latency_ms = int((time.time() - start_time) * 1000)
        
        # Track usage
        self.total_requests += 1
        self.total_tokens += response.usage.total_tokens
        cost = self.calculate_cost(response.usage)
        self.total_cost += cost
        
        logger.info(
            f"{self.model}: {response.usage.total_tokens} tokens, "
            f"${cost:.4f}, {response.latency_ms}ms"
        )
        
        return response
    
    @abstractmethod
    async def _complete_impl(
        self, 
        prompt: str, 
        response_format: Type[BaseModel],
        **kwargs
    ) -> LLMResponse:
        """
        Provider-specific implementation of completion.
        
        Must be implemented by each provider.
        """
        pass
    
    @abstractmethod
    def calculate_cost(self, usage: TokenUsage) -> float:
        """
        Calculate cost in USD from token usage.
        
        Must be implemented by each provider based on their pricing.
        """
        pass
    
    def get_stats(self) -> dict:
        """Get provider usage statistics."""
        return {
            "model": self.model,
            "requests": self.total_requests,
            "tokens": self.total_tokens,
            "cost": round(self.total_cost, 4),
            "avg_tokens_per_request": (
                self.total_tokens / max(1, self.total_requests)
            )
        }

