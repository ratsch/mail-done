"""
Apple Mail Integration Routes

Endpoints for syncing colors and metadata between Apple Mail and the database.
Used by AppleScript to look up email colors and apply them in Mail.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional, List
import logging

from backend.api.auth import verify_api_key
from backend.core.database import get_db
from backend.core.database.models import Email, EmailMetadata
from backend.core.email.models import AppleMailColor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/applemail", tags=["Apple Mail"])


# Request/Response Models

class ColorLookupRequest(BaseModel):
    """Request to look up color for a single email"""
    message_id: str = Field(..., description="RFC822 Message-ID header")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message_id": "<CABc1234567890@mail.gmail.com>"
            }
        }


class ColorLookupResponse(BaseModel):
    """Response with color information"""
    message_id: str
    color: str = Field(..., description="Color name: red, orange, yellow, green, blue, purple, gray, none")
    color_number: int = Field(..., description="Color number 0-7 for AppleScript")
    category: Optional[str] = Field(None, description="AI category if available")
    confidence: Optional[float] = Field(None, description="AI confidence 0-1")
    found: bool = Field(..., description="Whether email was found in database")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message_id": "<CABc1234@mail.gmail.com>",
                "color": "red",
                "color_number": 1,
                "category": "receipts.shopping.online",
                "confidence": 0.95,
                "found": True
            }
        }


class BatchColorLookupRequest(BaseModel):
    """Request to look up colors for multiple emails"""
    message_ids: List[str] = Field(..., description="List of RFC822 Message-IDs")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message_ids": [
                    "<CABc1234@mail.gmail.com>",
                    "<CABc5678@mail.gmail.com>"
                ]
            }
        }


class BatchColorLookupResponse(BaseModel):
    """Response with colors for multiple emails"""
    results: List[ColorLookupResponse]


class AppleMailStatsResponse(BaseModel):
    """Statistics about emails and colors in database"""
    total_emails: int
    emails_with_metadata: int
    emails_with_colors: int
    color_breakdown: dict
    recent_updates: int


# Helper Functions

COLOR_NAMES = {
    AppleMailColor.NONE: "none",
    AppleMailColor.RED: "red",
    AppleMailColor.ORANGE: "orange",
    AppleMailColor.YELLOW: "yellow",
    AppleMailColor.GREEN: "green",
    AppleMailColor.BLUE: "blue",
    AppleMailColor.PURPLE: "purple",
    AppleMailColor.GRAY: "gray"
}


def get_color_info(email: Email) -> dict:
    """Extract color information from email record"""
    if not email:
        return {
            "color": "none",
            "color_number": 0,
            "category": None,
            "confidence": None,
            "found": False
        }
    
    # Check if metadata exists and has a color
    if email.email_metadata and email.email_metadata.intended_color:
        color_num = email.email_metadata.intended_color
        color_name = COLOR_NAMES.get(color_num, "none")
        
        return {
            "color": color_name,
            "color_number": color_num,
            "category": email.email_metadata.ai_category,
            "confidence": email.email_metadata.ai_confidence,
            "found": True
        }
    else:
        # Email found but no color assigned
        return {
            "color": "none",
            "color_number": 0,
            "category": None,
            "confidence": None,
            "found": True
        }


# Routes

@router.post("/color", response_model=ColorLookupResponse, dependencies=[Depends(verify_api_key)])
async def lookup_color(
    request: ColorLookupRequest,
    db: Session = Depends(get_db)
):
    """
    Look up the assigned color for a single email by Message-ID.
    
    This endpoint is called by AppleScript to determine what color
    to apply to emails in Apple Mail based on database classifications.
    
    Returns:
    - Color name (for AppleScript: red, orange, yellow, green, blue, purple, gray, none)
    - Color number (0-7)
    - AI category and confidence if available
    """
    try:
        # Query email by message_id
        email = db.query(Email).filter(
            Email.message_id == request.message_id
        ).first()
        
        color_info = get_color_info(email)
        
        return ColorLookupResponse(
            message_id=request.message_id,
            **color_info
        )
        
    except Exception as e:
        logger.error(f"Error looking up color for {request.message_id}: {e}")
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred. Please try again.")


@router.post("/color/batch", response_model=BatchColorLookupResponse, dependencies=[Depends(verify_api_key)])
async def lookup_colors_batch(
    request: BatchColorLookupRequest,
    db: Session = Depends(get_db)
):
    """
    Look up colors for multiple emails at once (batch operation).
    
    More efficient than making individual requests when processing
    many emails at once (e.g., applying colors to entire mailbox).
    
    Returns:
    - Array of color information for each Message-ID
    """
    try:
        # Query all emails at once
        emails = db.query(Email).filter(
            Email.message_id.in_(request.message_ids)
        ).all()
        
        # Build lookup map
        email_map = {e.message_id: e for e in emails}
        
        # Build results maintaining order
        results = []
        for msg_id in request.message_ids:
            email = email_map.get(msg_id)
            color_info = get_color_info(email)
            
            results.append(ColorLookupResponse(
                message_id=msg_id,
                **color_info
            ))
        
        return BatchColorLookupResponse(results=results)
        
    except Exception as e:
        logger.error(f"Error in batch color lookup: {e}")
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred. Please try again.")


@router.get("/stats", response_model=AppleMailStatsResponse, dependencies=[Depends(verify_api_key)])
async def get_applemail_stats(db: Session = Depends(get_db)):
    """
    Get statistics about Apple Mail integration.
    
    Returns:
    - Total emails in database
    - How many have colors assigned
    - Breakdown by color
    """
    try:
        from sqlalchemy import func
        from datetime import datetime, timedelta
        
        # Total counts
        total_emails = db.query(Email).count()
        emails_with_metadata = db.query(EmailMetadata).count()
        emails_with_colors = db.query(EmailMetadata).filter(
            EmailMetadata.intended_color != None,
            EmailMetadata.intended_color != 0
        ).count()
        
        # Color breakdown
        color_counts = db.query(
            EmailMetadata.intended_color,
            func.count(EmailMetadata.id)
        ).filter(
            EmailMetadata.intended_color != None,
            EmailMetadata.intended_color != 0
        ).group_by(EmailMetadata.intended_color).all()
        
        color_breakdown = {}
        for color_num, count in color_counts:
            color_name = COLOR_NAMES.get(color_num, f"unknown_{color_num}")
            color_breakdown[color_name] = count
        
        # Recent updates (last 24 hours)
        yesterday = datetime.utcnow() - timedelta(days=1)
        recent_updates = db.query(EmailMetadata).filter(
            EmailMetadata.updated_at >= yesterday
        ).count()
        
        return AppleMailStatsResponse(
            total_emails=total_emails,
            emails_with_metadata=emails_with_metadata,
            emails_with_colors=emails_with_colors,
            color_breakdown=color_breakdown,
            recent_updates=recent_updates
        )
        
    except Exception as e:
        logger.error(f"Error getting Apple Mail stats: {e}")
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred. Please try again.")


@router.get("/health")
async def applemail_health_check():
    """
    Simple health check for Apple Mail integration.
    
    This endpoint can be called by AppleScript to verify the API is running
    before attempting to process emails.
    """
    return {
        "status": "healthy",
        "service": "apple-mail-integration",
        "version": "1.0",
        "endpoints": {
            "lookup_single": "POST /api/applemail/color",
            "lookup_batch": "POST /api/applemail/color/batch",
            "stats": "GET /api/applemail/stats"
        }
    }

