"""
Scope Definitions and Checking

Defines the permission scopes for the signed auth system.
Uses a hierarchical wildcard system similar to MQTT topics.

Scope format: "resource:action" or "resource:*" for all actions
Special scope "*" grants all permissions (admin only).
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Set, FrozenSet
import fnmatch
import logging

logger = logging.getLogger(__name__)


# Special scope that grants all permissions
SCOPE_ALL = "*"


class ScopeCategory(str, Enum):
    """Categories of scopes for grouping."""
    EMAILS = "emails"
    SEARCH = "search"
    STATS = "stats"
    APPLICATIONS = "applications"
    ATTACHMENTS = "attachments"
    AUTH = "auth"
    ADMIN = "admin"


@dataclass(frozen=True)
class Scope:
    """
    Represents a permission scope.
    
    Format: "category:action" (e.g., "emails:read", "search:*")
    
    Attributes:
        category: The resource category (emails, search, stats, etc.)
        action: The allowed action (read, write, *, etc.)
    """
    category: str
    action: str
    
    @classmethod
    def parse(cls, scope_str: str) -> "Scope":
        """
        Parse a scope string into a Scope object.
        
        Args:
            scope_str: Scope string like "emails:read" or "*"
            
        Returns:
            Scope object
            
        Raises:
            ValueError: If scope format is invalid
        """
        scope_str = scope_str.strip()
        
        if scope_str == SCOPE_ALL:
            return cls(category="*", action="*")
        
        if ":" not in scope_str:
            raise ValueError(f"Invalid scope format '{scope_str}': expected 'category:action'")
        
        parts = scope_str.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid scope format '{scope_str}'")
        
        category, action = parts
        if not category or not action:
            raise ValueError(f"Invalid scope format '{scope_str}': empty category or action")
        
        return cls(category=category, action=action)
    
    def __str__(self) -> str:
        if self.category == "*" and self.action == "*":
            return SCOPE_ALL
        return f"{self.category}:{self.action}"
    
    def matches(self, required: "Scope") -> bool:
        """
        Check if this scope grants the required permission.
        
        Wildcard matching rules:
        - "*" matches everything
        - "category:*" matches all actions in that category
        - "category:action" matches exactly
        
        Args:
            required: The scope that is required
            
        Returns:
            True if this scope grants the required permission
        """
        # Admin scope matches everything
        if self.category == "*" and self.action == "*":
            return True
        
        # Category must match (exact or wildcard)
        if self.category != "*" and self.category != required.category:
            return False
        
        # Action must match (exact or wildcard)
        if self.action != "*" and self.action != required.action:
            return False
        
        return True


def parse_scopes(scope_strings: List[str]) -> FrozenSet[Scope]:
    """
    Parse a list of scope strings into Scope objects.
    
    Args:
        scope_strings: List of scope strings
        
    Returns:
        Frozen set of Scope objects
    """
    return frozenset(Scope.parse(s) for s in scope_strings)


def check_scope(granted_scopes: FrozenSet[Scope], required: str) -> bool:
    """
    Check if any of the granted scopes allow the required permission.
    
    Args:
        granted_scopes: Set of scopes the client has
        required: The scope string that is required (e.g., "emails:read")
        
    Returns:
        True if permission is granted, False otherwise
        
    Example:
        >>> scopes = parse_scopes(["emails:*", "stats:read"])
        >>> check_scope(scopes, "emails:read")  # True
        >>> check_scope(scopes, "emails:write")  # True (wildcard)
        >>> check_scope(scopes, "stats:read")  # True
        >>> check_scope(scopes, "stats:write")  # False
    """
    required_scope = Scope.parse(required)
    
    for granted in granted_scopes:
        if granted.matches(required_scope):
            return True
    
    return False


def check_scope_list(granted_scopes: List[str], required: str) -> bool:
    """
    Convenience function that accepts list of strings.
    
    Args:
        granted_scopes: List of scope strings the client has
        required: The scope string that is required
        
    Returns:
        True if permission is granted
    """
    return check_scope(parse_scopes(granted_scopes), required)


# Predefined scope sets for common client types
LAPTOP_ADMIN_SCOPES = frozenset([Scope.parse("*")])

WEB_UI_SCOPES = parse_scopes([
    "emails:read",
    "search:*",
    "stats:read",
    "attachments:read",
])

V0_PORTAL_SCOPES = parse_scopes([
    "applications:read",
    "applications:write",
    "auth:*",
])
