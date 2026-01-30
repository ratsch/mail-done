"""
Email processing trigger endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Optional
import asyncio

from backend.api.auth import verify_api_key
from backend.api.schemas import TriggerProcessingRequest, ProcessingStatusResponse

router = APIRouter(prefix="/api/process", tags=["processing"])


# Global processing state (in production, use Redis)
processing_state = {
    "is_running": False,
    "current_status": None
}


@router.post("/trigger", dependencies=[Depends(verify_api_key)])
async def trigger_processing(
    request: TriggerProcessingRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger email processing in the background.
    
    **Note:** This is a simplified version. In production, you should:
    - Use a task queue (Celery, Redis Queue)
    - Run processing in a separate worker
    - Use proper job tracking
    
    **For now, this returns immediately and runs processing in background.**
    
    **Better approach:** Run `process_inbox.py` as a cron job or scheduled task.
    """
    if processing_state["is_running"]:
        raise HTTPException(
            status_code=409,
            detail="Processing already in progress. Please wait."
        )
    
    # TODO: Implement actual processing trigger
    # This would need to:
    # 1. Import EmailProcessingPipeline from process_inbox.py
    # 2. Get IMAP credentials from environment
    # 3. Run processing asynchronously
    # 4. Update processing_state
    
    return {
        "status": "not_implemented",
        "message": "Processing trigger endpoint not yet implemented. Please run process_inbox.py directly or via cron job.",
        "recommendation": "Use: poetry run python process_inbox.py --limit 100"
    }


@router.get("/status", dependencies=[Depends(verify_api_key)])
async def get_processing_status():
    """
    Get current processing status.
    
    Returns whether processing is running and current stats.
    """
    return {
        "is_running": processing_state["is_running"],
        "status": processing_state.get("current_status", "idle"),
        "message": "Processing status tracking not yet implemented. Run process_inbox.py directly to process emails."
    }

