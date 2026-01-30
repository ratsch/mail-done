"""
Cost tracking and analytics endpoints
"""
from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta

from backend.api.auth import verify_api_key
from backend.core.database import get_db
from backend.core.database.models import APIUsage, DailyAPIUsage

router = APIRouter(prefix="/api/costs", tags=["costs"])


@router.get("/summary", dependencies=[Depends(verify_api_key)])
async def get_cost_summary(db: Session = Depends(get_db)):
    """
    Quick cost summary for dashboard
    
    Returns today's, this month's, and total costs
    """
    # Total all-time cost
    total_cost = db.query(func.sum(APIUsage.cost_usd)).scalar() or 0.0
    
    # This month
    month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    cost_this_month = db.query(func.sum(APIUsage.cost_usd)).filter(
        APIUsage.timestamp >= month_start
    ).scalar() or 0.0
    
    # Today
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    cost_today = db.query(func.sum(APIUsage.cost_usd)).filter(
        APIUsage.timestamp >= today_start
    ).scalar() or 0.0
    
    return {
        "total": float(total_cost),
        "this_month": float(cost_this_month),
        "today": float(cost_today)
    }


@router.get("/overview", dependencies=[Depends(verify_api_key)])
async def get_cost_overview(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive cost overview with detailed breakdown and projections
    
    Returns:
    - Total costs for the period
    - Breakdown by model (GPT-4o, GPT-4o-mini, embeddings)
    - Breakdown by task (classification, embedding, search)
    - Daily trends
    - Cost projections (monthly, yearly)
    """
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    # Total costs in period
    total_cost = db.query(func.sum(APIUsage.cost_usd)).filter(
        APIUsage.timestamp >= cutoff_date
    ).scalar() or 0.0
    
    # Total tokens
    total_tokens = db.query(func.sum(APIUsage.total_tokens)).filter(
        APIUsage.timestamp >= cutoff_date
    ).scalar() or 0
    
    # Breakdown by model
    model_costs = db.query(
        APIUsage.model,
        func.sum(APIUsage.cost_usd).label('cost'),
        func.sum(APIUsage.total_tokens).label('tokens'),
        func.count(APIUsage.id).label('calls')
    ).filter(
        APIUsage.timestamp >= cutoff_date
    ).group_by(APIUsage.model).all()
    
    model_breakdown = [
        {
            "model": model,
            "cost": float(cost),
            "tokens": int(tokens),
            "calls": int(calls),
            "avg_cost_per_call": float(cost / calls) if calls > 0 else 0
        }
        for model, cost, tokens, calls in model_costs
    ]
    
    # Breakdown by task
    task_costs = db.query(
        APIUsage.task,
        func.sum(APIUsage.cost_usd).label('cost'),
        func.sum(APIUsage.total_tokens).label('tokens'),
        func.count(APIUsage.id).label('calls')
    ).filter(
        APIUsage.timestamp >= cutoff_date
    ).group_by(APIUsage.task).all()
    
    task_breakdown = [
        {
            "task": task,
            "cost": float(cost),
            "tokens": int(tokens),
            "calls": int(calls)
        }
        for task, cost, tokens, calls in task_costs
    ]
    
    # Daily aggregated costs (wrap in try/except in case table doesn't exist)
    daily_trend = []
    try:
        daily_costs = db.query(DailyAPIUsage).filter(
            DailyAPIUsage.usage_date >= cutoff_date.date()
        ).order_by(DailyAPIUsage.usage_date).all()
        
        daily_trend = [
            {
                "date": str(day.usage_date),
                "cost": float(day.total_cost_usd),
                "tokens": int(day.total_tokens),
                "calls": int(day.total_calls)  # Fixed: use total_calls not api_calls
            }
            for day in daily_costs
        ]
    except Exception as e:
        # Table might not exist yet, that's OK
        pass
    
    # Calculate projections
    avg_daily_cost = total_cost / days if days > 0 else 0
    projected_monthly = avg_daily_cost * 30
    projected_yearly = avg_daily_cost * 365
    
    # Recent activity
    last_24h = datetime.utcnow() - timedelta(hours=24)
    cost_24h = db.query(func.sum(APIUsage.cost_usd)).filter(
        APIUsage.timestamp >= last_24h
    ).scalar() or 0.0
    
    last_7d = datetime.utcnow() - timedelta(days=7)
    cost_7d = db.query(func.sum(APIUsage.cost_usd)).filter(
        APIUsage.timestamp >= last_7d
    ).scalar() or 0.0
    
    return {
        "period_days": days,
        "total_cost": float(total_cost),
        "total_tokens": int(total_tokens),
        "cost_last_24h": float(cost_24h),
        "cost_last_7d": float(cost_7d),
        "avg_daily_cost": float(avg_daily_cost),
        "projections": {
            "monthly": float(projected_monthly),
            "yearly": float(projected_yearly)
        },
        "breakdown_by_model": model_breakdown,
        "breakdown_by_task": task_breakdown,
        "daily_trend": daily_trend
    }


@router.get("/recent", dependencies=[Depends(verify_api_key)])
async def get_recent_usage(
    limit: int = Query(100, ge=1, le=1000, description="Number of recent API calls"),
    db: Session = Depends(get_db)
):
    """
    Get recent API usage events
    
    Useful for debugging and monitoring
    """
    recent = db.query(APIUsage).order_by(
        APIUsage.timestamp.desc()
    ).limit(limit).all()
    
    return {
        "count": len(recent),
        "usage": [
            {
                "timestamp": usage.timestamp.isoformat(),
                "model": usage.model,
                "task": usage.task,
                "tokens": usage.total_tokens,
                "cost": float(usage.cost_usd),
                "source": usage.source
            }
            for usage in recent
        ]
    }

