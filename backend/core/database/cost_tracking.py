"""
OpenAI API Cost Tracking

Tracks all OpenAI API usage in PostgreSQL with:
- Per-call tracking (model, task, tokens, cost)
- Daily aggregation (total cost per day)
- Model breakdown (GPT-4o vs GPT-4o-mini vs embeddings)
- Task breakdown (classification vs embedding vs search)
- Source tracking (CLI vs API)
"""
import logging
from datetime import datetime, date, timezone
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, text
import uuid
from uuid import UUID

from .models import APIUsage, DailyAPIUsage
from . import get_db

logger = logging.getLogger(__name__)


class CostTracker:
    """Track OpenAI API usage and costs."""
    
    def __init__(self, db: Session, source: str = "cli", use_separate_session: bool = True):
        """
        Initialize cost tracker.
        
        Args:
            db: Database session
            source: Source of API calls ('cli' or 'api')
            use_separate_session: If True, create independent session for cost tracking
                                 (prevents rollback if main transaction fails)
        """
        self.db = db
        self.source = source
        self.use_separate_session = use_separate_session
        self._own_session = None
    
    def log_usage(
        self,
        model: str,
        task: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        cost_usd: float,
        email_id: Optional[UUID] = None,
        context_data: Optional[Dict[str, Any]] = None
    ) -> Optional[APIUsage]:
        """Log API usage with atomic daily aggregation."""
        
        # Calculate cost if not provided
        if cost_usd is None:
            cost_usd = self.calculate_cost(model, prompt_tokens, completion_tokens)
        
        # Get a new database session
        session = next(get_db())
        try:
            usage = APIUsage(
                id=uuid.uuid4(),
                timestamp=datetime.now(timezone.utc),
                model=model,
                task=task,
                source=self.source,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                email_id=email_id,
                context_data=context_data or {}
            )
            
            if self.use_separate_session:
                # Add usage record to session
                session.add(usage)
                
                # Attempt daily aggregation in same transaction
                try:
                    self._update_daily_aggregate_atomic(
                        session,
                        date.today(),
                        model,
                        task,
                        total_tokens,
                        cost_usd
                    )
                    # Commit both usage record and daily aggregate together
                    session.commit()
                except Exception as error:
                    logger.warning(f"Cost tracking failed: {error}")
                    session.rollback()
                    return None
                finally:
                    session.close()
            else:
                session.add(usage)
                
                # Use atomic UPSERT for daily aggregation (no locks, fast)
                try:
                    self._update_daily_aggregate_atomic(
                        session,
                        date.today(),
                        model,
                        task,
                        total_tokens,
                        cost_usd
                    )
                except Exception as agg_error:
                    # Silently fail - usage record is what matters
                    logger.debug(f"Daily aggregate skipped: {agg_error}")
                
                session.commit()
            
            logger.debug(f"Successfully logged API usage: {model} ({total_tokens} tokens, ${cost_usd:.4f})")
            return usage
        finally:
            session.close()
    
    def _update_daily_aggregate_atomic(
        self,
        session: Session,
        usage_date: date,
        model: str,
        task: str,
        tokens: int,
        cost: float
    ):
        """
        Update daily aggregate using atomic PostgreSQL UPSERT.
        This prevents lock contention and timeouts during bulk processing.
        """
        try:
            # Use PostgreSQL's INSERT ... ON CONFLICT for atomic operation
            # Much faster than SELECT + UPDATE, no row locks
            sql = text("""
                INSERT INTO daily_api_usage (
                    id, usage_date, model, task, source, 
                    total_calls, total_tokens, total_cost_usd,
                    created_at, updated_at
                )
                VALUES (
                    gen_random_uuid(), :usage_date, :model, :task, :source,
                    1, :tokens, :cost,
                    NOW(), NOW()
                )
                ON CONFLICT (usage_date, model, task, source)
                DO UPDATE SET
                    total_calls = daily_api_usage.total_calls + 1,
                    total_tokens = daily_api_usage.total_tokens + :tokens,
                    total_cost_usd = daily_api_usage.total_cost_usd + :cost,
                    updated_at = NOW()
            """)
            
            session.execute(sql, {
                'usage_date': usage_date,
                'model': model,
                'task': task,
                'source': self.source,
                'tokens': tokens,
                'cost': cost
            })
            
        except Exception as e:
            # Don't propagate - this is optimization, usage record is what matters
            logger.debug(f"Atomic daily aggregate failed: {e}")
    
    def _update_daily_aggregate_in_session(
        self,
        session: Session,
        usage_date: date,
        model: str,
        task: str,
        tokens: int,
        cost: float
    ):
        """Legacy method - kept for backwards compatibility."""
        # Delegate to atomic version
        self._update_daily_aggregate_atomic(session, usage_date, model, task, tokens, cost)
    
    def get_daily_summary(self, day: Optional[date] = None) -> dict:
        """
        Get cost summary for a specific day.
        
        Args:
            day: Date to query (default: today)
        
        Returns:
            Dict with breakdown by model and task
        """
        target_date = day or date.today()
        
        records = self.db.query(DailyAPIUsage).filter(
            DailyAPIUsage.usage_date == target_date
        ).all()
        
        summary = {
            'date': target_date.isoformat(),
            'total_cost': 0.0,
            'total_tokens': 0,
            'total_calls': 0,
            'by_model': {},
            'by_task': {},
            'by_source': {}
        }
        
        for record in records:
            summary['total_cost'] += record.total_cost_usd
            summary['total_tokens'] += record.total_tokens
            summary['total_calls'] += record.total_calls
            
            # By model
            if record.model not in summary['by_model']:
                summary['by_model'][record.model] = {'cost': 0.0, 'tokens': 0, 'calls': 0}
            summary['by_model'][record.model]['cost'] += record.total_cost_usd
            summary['by_model'][record.model]['tokens'] += record.total_tokens
            summary['by_model'][record.model]['calls'] += record.total_calls
            
            # By task
            if record.task not in summary['by_task']:
                summary['by_task'][record.task] = {'cost': 0.0, 'tokens': 0, 'calls': 0}
            summary['by_task'][record.task]['cost'] += record.total_cost_usd
            summary['by_task'][record.task]['tokens'] += record.total_tokens
            summary['by_task'][record.task]['calls'] += record.total_calls
            
            # By source
            if record.source not in summary['by_source']:
                summary['by_source'][record.source] = {'cost': 0.0, 'tokens': 0, 'calls': 0}
            summary['by_source'][record.source]['cost'] += record.total_cost_usd
            summary['by_source'][record.source]['tokens'] += record.total_tokens
            summary['by_source'][record.source]['calls'] += record.total_calls
        
        return summary
    
    def get_monthly_summary(self, year: int, month: int) -> dict:
        """Get cost summary for entire month."""
        from datetime import date as date_class
        from calendar import monthrange
        
        start_date = date_class(year, month, 1)
        _, last_day = monthrange(year, month)
        end_date = date_class(year, month, last_day)
        
        records = self.db.query(DailyAPIUsage).filter(
            DailyAPIUsage.usage_date >= start_date,
            DailyAPIUsage.usage_date <= end_date
        ).all()
        
        summary = {
            'month': f"{year}-{month:02d}",
            'total_cost': 0.0,
            'total_tokens': 0,
            'total_calls': 0,
            'by_model': {},
            'by_task': {},
            'by_source': {},
            'by_day': {}
        }
        
        for record in records:
            day_key = record.usage_date.isoformat()
            
            summary['total_cost'] += record.total_cost_usd
            summary['total_tokens'] += record.total_tokens
            summary['total_calls'] += record.total_calls
            
            # By model
            if record.model not in summary['by_model']:
                summary['by_model'][record.model] = {'cost': 0.0, 'tokens': 0, 'calls': 0}
            summary['by_model'][record.model]['cost'] += record.total_cost_usd
            summary['by_model'][record.model]['tokens'] += record.total_tokens
            summary['by_model'][record.model]['calls'] += record.total_calls
            
            # By task
            if record.task not in summary['by_task']:
                summary['by_task'][record.task] = {'cost': 0.0, 'tokens': 0, 'calls': 0}
            summary['by_task'][record.task]['cost'] += record.total_cost_usd
            summary['by_task'][record.task]['tokens'] += record.total_tokens
            summary['by_task'][record.task]['calls'] += record.total_calls
            
            # By source
            if record.source not in summary['by_source']:
                summary['by_source'][record.source] = {'cost': 0.0, 'tokens': 0, 'calls': 0}
            summary['by_source'][record.source]['cost'] += record.total_cost_usd
            summary['by_source'][record.source]['tokens'] += record.total_tokens
            summary['by_source'][record.source]['calls'] += record.total_calls
            
            # By day
            if day_key not in summary['by_day']:
                summary['by_day'][day_key] = {'cost': 0.0, 'tokens': 0, 'calls': 0}
            summary['by_day'][day_key]['cost'] += record.total_cost_usd
            summary['by_day'][day_key]['tokens'] += record.total_tokens
            summary['by_day'][day_key]['calls'] += record.total_calls
        
        return summary
    
    def get_cost_report(self, days: int = 30) -> str:
        """
        Generate a formatted cost report for the last N days.
        
        Args:
            days: Number of days to include
        
        Returns:
            Formatted report string
        """
        from datetime import timedelta
        
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        records = self.db.query(DailyAPIUsage).filter(
            DailyAPIUsage.usage_date >= start_date,
            DailyAPIUsage.usage_date <= end_date
        ).all()
        
        if not records:
            return f"No API usage recorded for the last {days} days."
        
        # Aggregate
        total_cost = sum(r.total_cost_usd for r in records)
        total_tokens = sum(r.total_tokens for r in records)
        total_calls = sum(r.total_calls for r in records)
        
        by_model = {}
        by_task = {}
        
        for r in records:
            by_model[r.model] = by_model.get(r.model, 0) + r.total_cost_usd
            by_task[r.task] = by_task.get(r.task, 0) + r.total_cost_usd
        
        # Build report
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      OPENAI API COST REPORT                                   ║
║                      Last {days} Days ({start_date} to {end_date})                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 OVERALL USAGE:
   Total API Calls: {total_calls:,}
   Total Tokens: {total_tokens:,}
   Total Cost: ${total_cost:.4f}
   
   Avg Cost/Call: ${total_cost / total_calls:.4f}
   Avg Tokens/Call: {total_tokens // total_calls:,}

💰 BY MODEL:
"""
        for model, cost in sorted(by_model.items(), key=lambda x: -x[1]):
            pct = (cost / total_cost * 100) if total_cost > 0 else 0
            report += f"   {model:30s}: ${cost:8.4f} ({pct:5.1f}%)\n"
        
        report += f"\n🎯 BY TASK:\n"
        for task, cost in sorted(by_task.items(), key=lambda x: -x[1]):
            pct = (cost / total_cost * 100) if total_cost > 0 else 0
            report += f"   {task:30s}: ${cost:8.4f} ({pct:5.1f}%)\n"
        
        report += f"\n📅 DAILY BREAKDOWN (Last 7 Days):\n"
        daily_totals = {}
        for r in records:
            day_key = r.usage_date.isoformat()
            daily_totals[day_key] = daily_totals.get(day_key, 0) + r.total_cost_usd
        
        for day in sorted(daily_totals.keys(), reverse=True)[:7]:
            report += f"   {day}: ${daily_totals[day]:.4f}\n"
        
        report += f"\n{'═'*80}\n"
        
        return report
    
    def commit(self):
        """Commit tracked usage to database."""
        try:
            self.db.commit()
        except Exception as e:
            logger.error(f"Failed to commit API usage: {e}")
            self.db.rollback()

