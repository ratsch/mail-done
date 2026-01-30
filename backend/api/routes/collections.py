"""
Application Collections Routes

Endpoints for managing collections of applications.
Collections allow users to organize applications into groups like
"Shortlist for interview", "Discuss in meeting", etc.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime
import logging

from backend.core.database import get_db
from backend.core.database.models import ApplicationCollection, ApplicationCollectionItem, Email, EmailMetadata
from backend.api.review_auth import get_current_reviewer, get_current_reviewer_hybrid
from backend.core.database.models import LabMember

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/collections", tags=["collections"])


# =============================================================================
# Pydantic Schemas
# =============================================================================

class CollectionCreate(BaseModel):
    """Request to create a new collection"""
    name: str = Field(..., min_length=1, max_length=255, description="Collection name")


class CollectionResponse(BaseModel):
    """Collection response with item count"""
    id: UUID
    name: str
    created_at: datetime
    item_count: int
    
    class Config:
        from_attributes = True


class CollectionItemsRequest(BaseModel):
    """Request to add/remove items from a collection"""
    email_ids: List[UUID] = Field(..., min_items=1, description="List of application email IDs")


class CollectionItemResponse(BaseModel):
    """Response for collection item operations"""
    success: bool
    added_count: int = 0
    removed_count: int = 0
    already_exists_count: int = 0
    not_found_count: int = 0


# =============================================================================
# Collection CRUD Endpoints
# =============================================================================

@router.get("", response_model=List[CollectionResponse])
async def list_collections(
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db)
):
    """
    List all collections with their item counts.
    
    Returns collections sorted by name.
    """
    try:
        # Query collections with item counts using a subquery
        collections = db.query(
            ApplicationCollection,
            func.count(ApplicationCollectionItem.id).label('item_count')
        ).outerjoin(
            ApplicationCollectionItem,
            ApplicationCollection.id == ApplicationCollectionItem.collection_id
        ).group_by(
            ApplicationCollection.id
        ).order_by(
            ApplicationCollection.name
        ).all()
        
        return [
            CollectionResponse(
                id=collection.id,
                name=collection.name,
                created_at=collection.created_at,
                item_count=item_count
            )
            for collection, item_count in collections
        ]
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        raise HTTPException(status_code=500, detail="Failed to list collections")


@router.post("", response_model=CollectionResponse, status_code=status.HTTP_201_CREATED)
async def create_collection(
    request: CollectionCreate,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db)
):
    """
    Create a new collection.
    
    Collection names must be unique.
    """
    try:
        # Check if collection with this name already exists
        existing = db.query(ApplicationCollection).filter(
            ApplicationCollection.name == request.name.strip()
        ).first()
        
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Collection '{request.name}' already exists"
            )
        
        # Create new collection
        collection = ApplicationCollection(
            name=request.name.strip()
        )
        db.add(collection)
        db.commit()
        db.refresh(collection)
        
        logger.info(f"Collection '{collection.name}' created by {current_user.email}")
        
        return CollectionResponse(
            id=collection.id,
            name=collection.name,
            created_at=collection.created_at,
            item_count=0
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create collection")


@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_collection(
    collection_id: UUID,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db)
):
    """
    Delete a collection.
    
    This removes the collection and all its item associations.
    The applications themselves are not affected.
    """
    try:
        collection = db.query(ApplicationCollection).filter(
            ApplicationCollection.id == collection_id
        ).first()
        
        if not collection:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Collection not found"
            )
        
        collection_name = collection.name
        db.delete(collection)
        db.commit()
        
        logger.info(f"Collection '{collection_name}' deleted by {current_user.email}")
        
        return None
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting collection: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete collection")


# =============================================================================
# Collection Items Endpoints
# =============================================================================

@router.post("/{collection_id}/items", response_model=CollectionItemResponse)
async def add_items_to_collection(
    collection_id: UUID,
    request: CollectionItemsRequest,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db)
):
    """
    Add applications to a collection.
    
    Applications that are already in the collection are skipped.
    """
    try:
        # Verify collection exists
        collection = db.query(ApplicationCollection).filter(
            ApplicationCollection.id == collection_id
        ).first()
        
        if not collection:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Collection not found"
            )
        
        added_count = 0
        already_exists_count = 0
        not_found_count = 0
        
        for email_id in request.email_ids:
            # Verify the email exists and is an application
            email = db.query(Email).join(
                EmailMetadata, Email.id == EmailMetadata.email_id
            ).filter(
                Email.id == email_id,
                EmailMetadata.ai_category.like('application-%')
            ).first()
            
            if not email:
                not_found_count += 1
                continue
            
            # Check if already in collection
            existing_item = db.query(ApplicationCollectionItem).filter(
                ApplicationCollectionItem.collection_id == collection_id,
                ApplicationCollectionItem.email_id == email_id
            ).first()
            
            if existing_item:
                already_exists_count += 1
                continue
            
            # Add to collection
            item = ApplicationCollectionItem(
                collection_id=collection_id,
                email_id=email_id
            )
            db.add(item)
            added_count += 1
        
        db.commit()
        
        logger.info(f"Added {added_count} items to collection '{collection.name}' by {current_user.email}")
        
        return CollectionItemResponse(
            success=True,
            added_count=added_count,
            already_exists_count=already_exists_count,
            not_found_count=not_found_count
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding items to collection: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to add items to collection")


@router.delete("/{collection_id}/items", response_model=CollectionItemResponse)
async def remove_items_from_collection(
    collection_id: UUID,
    request: CollectionItemsRequest,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db)
):
    """
    Remove applications from a collection.
    
    Applications that are not in the collection are skipped.
    """
    try:
        # Verify collection exists
        collection = db.query(ApplicationCollection).filter(
            ApplicationCollection.id == collection_id
        ).first()
        
        if not collection:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Collection not found"
            )
        
        removed_count = 0
        not_found_count = 0
        
        for email_id in request.email_ids:
            # Find the item
            item = db.query(ApplicationCollectionItem).filter(
                ApplicationCollectionItem.collection_id == collection_id,
                ApplicationCollectionItem.email_id == email_id
            ).first()
            
            if not item:
                not_found_count += 1
                continue
            
            db.delete(item)
            removed_count += 1
        
        db.commit()
        
        logger.info(f"Removed {removed_count} items from collection '{collection.name}' by {current_user.email}")
        
        return CollectionItemResponse(
            success=True,
            removed_count=removed_count,
            not_found_count=not_found_count
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing items from collection: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to remove items from collection")


@router.get("/{collection_id}/items", response_model=List[UUID])
async def get_collection_items(
    collection_id: UUID,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db)
):
    """
    Get all application IDs in a collection.
    
    Returns a list of email_ids.
    """
    try:
        # Verify collection exists
        collection = db.query(ApplicationCollection).filter(
            ApplicationCollection.id == collection_id
        ).first()
        
        if not collection:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Collection not found"
            )
        
        items = db.query(ApplicationCollectionItem.email_id).filter(
            ApplicationCollectionItem.collection_id == collection_id
        ).all()
        
        return [item.email_id for item in items]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting collection items: {e}")
        raise HTTPException(status_code=500, detail="Failed to get collection items")
