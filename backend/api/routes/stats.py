"""
Statistics and analytics API endpoints
"""
from typing import Optional, List
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_
from datetime import datetime, timedelta

from backend.api.auth import verify_api_key
from backend.api.schemas import StatsResponse, SenderStatsResponse
from backend.core.database import get_db
from backend.core.database.models import Email, EmailMetadata, Classification, SenderHistory, EmailEmbedding

router = APIRouter(prefix="/api/stats", tags=["statistics"])


@router.get("", response_model=StatsResponse, dependencies=[Depends(verify_api_key)])
async def get_system_stats(db: Session = Depends(get_db)):
    """
    Get overall system statistics.
    
    **Returns:**
    - Total emails in database
    - Emails today/this week
    - VIP and rule counts
    - AI classification count
    - Needs reply / flagged / unread counts
    - Top senders
    - Category breakdown
    """
    # Total emails
    total_emails = db.query(func.count(Email.id)).scalar() or 0
    
    # Emails today
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    emails_today = db.query(func.count(Email.id)).filter(
        Email.date >= today_start
    ).scalar() or 0
    
    # Emails this week
    week_start = today_start - timedelta(days=today_start.weekday())
    emails_this_week = db.query(func.count(Email.id)).filter(
        Email.date >= week_start
    ).scalar() or 0
    
    # Needs reply count
    needs_reply_count = db.query(func.count(EmailMetadata.id)).filter(
        EmailMetadata.needs_reply == True
    ).scalar() or 0
    
    # Flagged count
    flagged_count = db.query(func.count(Email.id)).filter(
        Email.is_flagged == True
    ).scalar() or 0
    
    # Unread count
    unread_count = db.query(func.count(Email.id)).filter(
        Email.is_seen == False
    ).scalar() or 0
    
    # AI classifications count
    ai_classifications = db.query(func.count(Classification.id)).filter(
        Classification.classifier_type == 'ai'
    ).scalar() or 0
    
    # Embeddings count
    embeddings_count = db.query(func.count(EmailEmbedding.id)).scalar() or 0
    
    # Top senders (top 10 by email count)
    top_senders = db.query(
        Email.from_address,
        func.max(SenderHistory.sender_name).label('sender_name'),
        func.count(Email.id).label('count')
    ).join(
        SenderHistory,
        Email.from_address == SenderHistory.email_address,
        isouter=True
    ).group_by(
        Email.from_address
    ).order_by(
        desc('count')
    ).limit(10).all()
    
    top_senders_list = [
        {
            "email": email,
            "name": name,
            "count": count
        }
        for email, name, count in top_senders
    ]
    
    # Category breakdown
    categories = db.query(
        EmailMetadata.ai_category,
        func.count(EmailMetadata.id).label('count')
    ).filter(
        EmailMetadata.ai_category.isnot(None)
    ).group_by(
        EmailMetadata.ai_category
    ).all()
    
    categories_breakdown = {cat: count for cat, count in categories if cat}
    
    # Folder breakdown
    folders = db.query(
        Email.folder,
        func.count(Email.id).label('count')
    ).group_by(
        Email.folder
    ).all()
    
    folders_breakdown = {folder: count for folder, count in folders if folder}
    
    # VIPs and rules (from config - hardcoded for now)
    vips_configured = 15  # TODO: Read from config
    rules_configured = 22  # TODO: Read from config
    
    return StatsResponse(
        total_emails=total_emails,
        emails_today=emails_today,
        emails_this_week=emails_this_week,
        vips_configured=vips_configured,
        rules_configured=rules_configured,
        ai_classifications=ai_classifications,
        needs_reply_count=needs_reply_count,
        flagged_count=flagged_count,
        unread_count=unread_count,
        top_senders=top_senders_list,
        categories_breakdown=categories_breakdown,
        folders_breakdown=folders_breakdown,
        with_embeddings=embeddings_count  # Add embeddings count
    )


@router.get("/senders", dependencies=[Depends(verify_api_key)])
async def list_senders(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    sort_by: str = Query("email_count", description="Sort by: email_count, last_seen, sender_name"),
    db: Session = Depends(get_db)
):
    """
    List all senders with statistics.
    
    **Sort options:**
    - email_count: Most frequent senders
    - last_seen: Most recent
    - sender_name: Alphabetical
    """
    query = db.query(SenderHistory)
    
    # Apply sorting
    if sort_by == "email_count":
        query = query.order_by(desc(SenderHistory.email_count))
    elif sort_by == "last_seen":
        query = query.order_by(desc(SenderHistory.last_seen))
    elif sort_by == "sender_name":
        query = query.order_by(SenderHistory.sender_name)
    
    # Get total
    total = query.count()
    
    # Paginate
    offset = (page - 1) * page_size
    senders = query.offset(offset).limit(page_size).all()
    
    # Calculate pages
    pages = (total + page_size - 1) // page_size
    
    return {
        "senders": senders,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": pages
    }


@router.get("/senders/{email_address:path}", response_model=SenderStatsResponse, dependencies=[Depends(verify_api_key)])
async def get_sender_stats(
    email_address: str,
    db: Session = Depends(get_db)
):
    """
    Get detailed statistics for a specific sender.
    
    **Returns:**
    - Email count
    - Last seen date
    - Sender type
    - Category breakdown
    - Average reply time
    """
    sender = db.query(SenderHistory).filter(
        SenderHistory.email_address == email_address
    ).first()
    
    if not sender:
        # Create basic response from emails
        emails = db.query(Email).filter(Email.from_address == email_address).all()
        if not emails:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Sender not found")
        
        email_count = len(emails)
        last_seen = max(e.date for e in emails) if emails else None
        
        # Get category breakdown
        categories_query = db.query(
            EmailMetadata.ai_category,
            func.count(EmailMetadata.id).label('count')
        ).join(Email, EmailMetadata.email_id == Email.id).filter(
            Email.from_address == email_address,
            EmailMetadata.ai_category.isnot(None)
        ).group_by(EmailMetadata.ai_category).all()
        
        categories = {cat: count for cat, count in categories_query if cat}
        
        return SenderStatsResponse(
            email_address=email_address,
            sender_name=emails[0].from_address if emails else None,
            email_count=email_count,
            last_seen=last_seen,
            sender_type=None,
            categories=categories,
            avg_reply_time_hours=None
        )
    
    # Get category breakdown for this sender
    categories_query = db.query(
        EmailMetadata.ai_category,
        func.count(EmailMetadata.id).label('count')
    ).join(Email, EmailMetadata.email_id == Email.id).filter(
        Email.from_address == email_address,
        EmailMetadata.ai_category.isnot(None)
    ).group_by(EmailMetadata.ai_category).all()
    
    categories = {cat: count for cat, count in categories_query if cat}
    
    return SenderStatsResponse(
        email_address=sender.email_address,
        sender_name=sender.sender_name,
        email_count=sender.email_count,
        last_seen=sender.last_seen,
        sender_type=sender.sender_type,
        categories=categories,
        avg_reply_time_hours=sender.avg_reply_time_hours
    )


@router.get("/categories/breakdown", dependencies=[Depends(verify_api_key)])
async def get_category_breakdown(
    time_range: Optional[str] = Query("all", description="all, today, week, month"),
    db: Session = Depends(get_db)
):
    """
    Get email count breakdown by category.
    
    **Time ranges:**
    - all: All emails
    - today: Last 24 hours
    - week: Last 7 days
    - month: Last 30 days
    """
    query = db.query(
        EmailMetadata.ai_category,
        func.count(EmailMetadata.id).label('count')
    ).join(Email, EmailMetadata.email_id == Email.id)
    
    # Apply time filter
    if time_range == "today":
        cutoff = datetime.utcnow() - timedelta(days=1)
        query = query.filter(Email.date >= cutoff)
    elif time_range == "week":
        cutoff = datetime.utcnow() - timedelta(days=7)
        query = query.filter(Email.date >= cutoff)
    elif time_range == "month":
        cutoff = datetime.utcnow() - timedelta(days=30)
        query = query.filter(Email.date >= cutoff)
    
    categories = query.filter(
        EmailMetadata.ai_category.isnot(None)
    ).group_by(EmailMetadata.ai_category).all()
    
    return {
        "time_range": time_range,
        "categories": {cat: count for cat, count in categories if cat}
    }

