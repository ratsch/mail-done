"""
Retry Manager - Exponential Backoff for Cross-Account Operations

Manages retry logic with exponential backoff and permanent error detection.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Callable, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class RetryManager:
    """
    Manages retry logic for cross-account operations with exponential backoff.
    
    Provides automatic retry with exponential backoff (2s, 4s, 8s...) and
    permanent error detection to avoid retrying unrecoverable failures.
    """
    
    def __init__(self, max_retries: int = 3, base_delay: float = 2.0):
        """
        Initialize retry manager.
        
        Args:
            max_retries: Maximum number of retry attempts (default: 3)
            base_delay: Base delay in seconds for exponential backoff (default: 2.0)
                       Results in delays of: 2s, 4s, 8s, 16s...
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    async def execute_with_retry(self,
                                 operation: Callable,
                                 operation_name: str,
                                 *args,
                                 **kwargs) -> Tuple[bool, Optional[str]]:
        """
        Execute an operation with automatic retry on failure.
        
        The operation should be an async function that returns (success: bool, error: Optional[str]).
        
        Args:
            operation: Async callable that returns (success, error) tuple
            operation_name: Name of operation for logging
            *args, **kwargs: Arguments to pass to operation
        
        Returns:
            (success: bool, error_message: Optional[str])
        """
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    # Exponential backoff: 2^attempt * base_delay
                    delay = (2 ** (attempt - 1)) * self.base_delay
                    logger.info(f"Retry attempt {attempt}/{self.max_retries} for {operation_name} "
                               f"after {delay}s delay...")
                    await asyncio.sleep(delay)
                
                # Execute the operation
                result = await operation(*args, **kwargs)
                
                # Handle both tuple and single return value
                if isinstance(result, tuple) and len(result) == 2:
                    success, error = result
                else:
                    # Assume single boolean means success
                    success = bool(result)
                    error = None if success else "Operation failed"
                
                if success:
                    if attempt > 0:
                        logger.info(f"{operation_name} succeeded on attempt {attempt + 1}")
                    return True, None
                
                # Log failure but continue retrying
                logger.warning(f"{operation_name} failed on attempt {attempt + 1}: {error}")
                
                # Don't retry certain errors (e.g., permission denied)
                if self._is_permanent_error(error):
                    logger.error(f"Permanent error detected, stopping retries: {error}")
                    return False, f"Permanent error: {error}"
                
            except Exception as e:
                logger.error(f"Exception in {operation_name} attempt {attempt + 1}: {e}")
                if attempt == self.max_retries:
                    return False, f"Max retries exceeded: {str(e)}"
                # Check if exception indicates permanent error
                if self._is_permanent_error(str(e)):
                    return False, f"Permanent error: {str(e)}"
        
        return False, f"Failed after {self.max_retries + 1} attempts"
    
    def _is_permanent_error(self, error: Optional[str]) -> bool:
        """
        Check if error is permanent (should not retry).
        
        Args:
            error: Error message string
        
        Returns:
            True if error is permanent and should not be retried
        """
        if not error:
            return False
        
        permanent_errors = [
            'permission denied',
            'authentication failed',
            'invalid credentials',
            'folder not allowed',
            'not in whitelist',
            'invalid folder path',
            'account not found',
            'move not allowed',
            'authentication error',
            'unauthorized',
            'forbidden',
            '403',
            '401'
        ]
        
        error_lower = error.lower()
        return any(perm_err in error_lower for perm_err in permanent_errors)
    
    def calculate_next_retry(self, attempt: int) -> datetime:
        """
        Calculate next retry timestamp based on attempt number.
        
        Args:
            attempt: Current attempt number (0-indexed)
        
        Returns:
            datetime for next retry attempt
        """
        delay = (2 ** attempt) * self.base_delay
        return datetime.utcnow() + timedelta(seconds=delay)
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay in seconds for given attempt number.
        
        Args:
            attempt: Attempt number (0-indexed)
        
        Returns:
            Delay in seconds
        """
        return (2 ** attempt) * self.base_delay

