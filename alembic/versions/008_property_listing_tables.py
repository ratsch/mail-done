"""Add property listing tables

Revision ID: 5348f1936c09
Revises: 007_review_assignments
Create Date: 2026-03-22 23:18:47.712840

Adds 10 tables for the real estate portal:
  property_listings, property_listing_sources, property_reviews,
  property_actions, property_private_notes, property_due_diligence,
  property_documents, property_collections, property_collection_items,
  property_share_tokens
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '5348f1936c09'
down_revision: Union[str, Sequence[str], None] = '007_review_assignments'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- Parent tables (no FK to other new tables) ---

    op.create_table('property_collections',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('created_by', sa.UUID(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['created_by'], ['lab_members.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name')
    )

    op.create_table('property_listings',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('email_id', sa.UUID(), nullable=True),
    sa.Column('address', sa.String(), nullable=True),
    sa.Column('plz', sa.String(), nullable=True),
    sa.Column('municipality', sa.String(), nullable=True),
    sa.Column('canton', sa.String(), nullable=True),
    sa.Column('listing_url', sa.String(), nullable=True),
    sa.Column('listing_source', sa.String(), nullable=True),
    sa.Column('listing_ref_id', sa.String(), nullable=True),
    sa.Column('dedup_hash', sa.String(), nullable=True),
    sa.Column('property_type', sa.String(), nullable=True),
    sa.Column('price_chf', sa.Integer(), nullable=True),
    sa.Column('price_known', sa.Boolean(), nullable=True),
    sa.Column('living_area_sqm', sa.Integer(), nullable=True),
    sa.Column('land_area_sqm', sa.Integer(), nullable=True),
    sa.Column('rooms', sa.Float(), nullable=True),
    sa.Column('year_built', sa.Integer(), nullable=True),
    sa.Column('last_renovation', sa.Integer(), nullable=True),
    sa.Column('price_per_sqm', sa.Integer(), nullable=True),
    sa.Column('floor', sa.Integer(), nullable=True),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('heating_type', sa.String(), nullable=True),
    sa.Column('heating_cost_yearly', sa.Integer(), nullable=True),
    sa.Column('parking_spaces', sa.Integer(), nullable=True),
    sa.Column('parking_included', sa.Boolean(), nullable=True),
    sa.Column('num_units_in_building', sa.Integer(), nullable=True),
    sa.Column('wertquote', sa.String(), nullable=True),
    sa.Column('erneuerungsfonds_chf', sa.Integer(), nullable=True),
    sa.Column('nebenkosten_yearly', sa.Integer(), nullable=True),
    sa.Column('zweitwohnung_allowed', sa.Boolean(), nullable=True),
    sa.Column('has_mountain_view', sa.Boolean(), nullable=True),
    sa.Column('has_lake_view', sa.Boolean(), nullable=True),
    sa.Column('has_garden_access', sa.Boolean(), nullable=True),
    sa.Column('has_terrace', sa.Boolean(), nullable=True),
    sa.Column('is_zweitwohnung', sa.Boolean(), nullable=True),
    sa.Column('is_baurecht', sa.Boolean(), nullable=True),
    sa.Column('is_stockwerkeigentum', sa.Boolean(), nullable=True),
    sa.Column('macro_location_score', sa.Integer(), nullable=True),
    sa.Column('micro_location_score', sa.Integer(), nullable=True),
    sa.Column('property_quality_score', sa.Integer(), nullable=True),
    sa.Column('garden_outdoor_score', sa.Integer(), nullable=True),
    sa.Column('financial_score', sa.Integer(), nullable=True),
    sa.Column('overall_recommendation', sa.Integer(), nullable=True),
    sa.Column('applicable_scenarios', sa.JSON(), nullable=True),
    sa.Column('scenario_scores', sa.JSON(), nullable=True),
    sa.Column('best_scenario', sa.String(), nullable=True),
    sa.Column('houzy_property_id', sa.String(), nullable=True),
    sa.Column('houzy_min', sa.Integer(), nullable=True),
    sa.Column('houzy_mid', sa.Integer(), nullable=True),
    sa.Column('houzy_max', sa.Integer(), nullable=True),
    sa.Column('houzy_quality_pct', sa.Integer(), nullable=True),
    sa.Column('price_vs_houzy_pct', sa.Float(), nullable=True),
    sa.Column('houzy_assessment', sa.String(), nullable=True),
    sa.Column('zustand_rating', sa.Float(), nullable=True),
    sa.Column('zustand_confirmed', sa.Boolean(), nullable=True),
    sa.Column('ausbaustandard_rating', sa.Float(), nullable=True),
    sa.Column('ausbaustandard_confirmed', sa.Boolean(), nullable=True),
    sa.Column('houzy_location_scores', sa.JSON(), nullable=True),
    sa.Column('houzy_fetched_at', sa.DateTime(), nullable=True),
    sa.Column('tier', sa.Integer(), nullable=True),
    sa.Column('completeness_pct', sa.Integer(), nullable=True),
    sa.Column('missing_fields', sa.JSON(), nullable=True),
    sa.Column('estimated_renovation_low', sa.Integer(), nullable=True),
    sa.Column('estimated_renovation_high', sa.Integer(), nullable=True),
    sa.Column('estimated_nebenkosten', sa.Integer(), nullable=True),
    sa.Column('highlights', sa.JSON(), nullable=True),
    sa.Column('red_flags', sa.JSON(), nullable=True),
    sa.Column('property_tags', sa.JSON(), nullable=True),
    sa.Column('ai_reasoning', sa.Text(), nullable=True),
    sa.Column('ai_full_response', sa.JSON(), nullable=True),
    sa.Column('ai_model', sa.String(), nullable=True),
    sa.Column('ai_scored_at', sa.DateTime(), nullable=True),
    sa.Column('outcome', sa.String(), nullable=True),
    sa.Column('outcome_notes', sa.Text(), nullable=True),
    sa.Column('outcome_date', sa.DateTime(), nullable=True),
    sa.Column('listing_status', sa.String(), nullable=True),
    sa.Column('first_seen', sa.DateTime(), nullable=True),
    sa.Column('days_on_market', sa.Integer(), nullable=True),
    sa.Column('price_history', sa.JSON(), nullable=True),
    sa.Column('agent_name', sa.String(), nullable=True),
    sa.Column('agent_email', sa.String(), nullable=True),
    sa.Column('agent_phone', sa.String(), nullable=True),
    sa.Column('agent_company', sa.String(), nullable=True),
    sa.Column('last_contacted_at', sa.DateTime(), nullable=True),
    sa.Column('contact_method', sa.String(), nullable=True),
    sa.Column('latitude', sa.Float(), nullable=True),
    sa.Column('longitude', sa.Float(), nullable=True),
    sa.Column('geocoded', sa.Boolean(), nullable=True),
    sa.Column('photo_urls', sa.JSON(), nullable=True),
    sa.Column('photos_archived', sa.Boolean(), nullable=True),
    sa.Column('gdrive_folder_id', sa.String(), nullable=True),
    sa.Column('gdrive_folder_url', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.Column('archived_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['email_id'], ['emails.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('dedup_hash')
    )
    op.create_index('ix_property_listings_dedup', 'property_listings', ['dedup_hash'], unique=False)
    op.create_index('ix_property_listings_plz', 'property_listings', ['plz'], unique=False)
    op.create_index('ix_property_listings_price', 'property_listings', ['price_chf'], unique=False)
    op.create_index('ix_property_listings_recommendation', 'property_listings', ['overall_recommendation'], unique=False)
    op.create_index('ix_property_listings_source', 'property_listings', ['listing_source'], unique=False)
    op.create_index('ix_property_listings_status', 'property_listings', ['listing_status'], unique=False)
    op.create_index('ix_property_listings_tier', 'property_listings', ['tier'], unique=False)
    op.create_index('ix_property_listings_type', 'property_listings', ['property_type'], unique=False)

    # --- Child tables (FK to property_listings and/or property_collections) ---

    op.create_table('property_actions',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('listing_id', sa.UUID(), nullable=False),
    sa.Column('acted_by', sa.UUID(), nullable=False),
    sa.Column('action', sa.String(), nullable=False),
    sa.Column('notes', sa.Text(), nullable=True),
    sa.Column('resulting_status', sa.String(), nullable=True),
    sa.Column('acted_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['acted_by'], ['lab_members.id'], ),
    sa.ForeignKeyConstraint(['listing_id'], ['property_listings.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_property_actions_acted_at', 'property_actions', ['acted_at'], unique=False)
    op.create_index('ix_property_actions_listing', 'property_actions', ['listing_id'], unique=False)

    op.create_table('property_collection_items',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('collection_id', sa.UUID(), nullable=False),
    sa.Column('listing_id', sa.UUID(), nullable=False),
    sa.Column('added_at', sa.DateTime(), nullable=True),
    sa.Column('notes', sa.Text(), nullable=True),
    sa.ForeignKeyConstraint(['collection_id'], ['property_collections.id'], ),
    sa.ForeignKeyConstraint(['listing_id'], ['property_listings.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('collection_id', 'listing_id', name='uq_collection_listing')
    )

    op.create_table('property_documents',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('listing_id', sa.UUID(), nullable=False),
    sa.Column('document_type', sa.String(), nullable=False),
    sa.Column('filename', sa.String(), nullable=False),
    sa.Column('gdrive_file_id', sa.String(), nullable=True),
    sa.Column('gdrive_link', sa.String(), nullable=True),
    sa.Column('local_path', sa.String(), nullable=True),
    sa.Column('source', sa.String(), nullable=True),
    sa.Column('uploaded_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['listing_id'], ['property_listings.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_property_documents_listing', 'property_documents', ['listing_id'], unique=False)

    op.create_table('property_due_diligence',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('listing_id', sa.UUID(), nullable=False),
    sa.Column('category', sa.String(), nullable=False),
    sa.Column('question', sa.String(), nullable=False),
    sa.Column('priority', sa.String(), nullable=True),
    sa.Column('status', sa.String(), nullable=True),
    sa.Column('answer', sa.Text(), nullable=True),
    sa.Column('source', sa.String(), nullable=True),
    sa.Column('date_answered', sa.DateTime(), nullable=True),
    sa.Column('is_deal_breaker', sa.Boolean(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['listing_id'], ['property_listings.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_property_due_diligence_listing', 'property_due_diligence', ['listing_id'], unique=False)

    op.create_table('property_listing_sources',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('listing_id', sa.UUID(), nullable=False),
    sa.Column('email_id', sa.UUID(), nullable=True),
    sa.Column('source', sa.String(), nullable=False),
    sa.Column('listing_url', sa.String(), nullable=True),
    sa.Column('price_at_source', sa.Integer(), nullable=True),
    sa.Column('first_seen_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['email_id'], ['emails.id'], ),
    sa.ForeignKeyConstraint(['listing_id'], ['property_listings.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_property_listing_sources_email', 'property_listing_sources', ['email_id'], unique=False)
    op.create_index('ix_property_listing_sources_listing', 'property_listing_sources', ['listing_id'], unique=False)

    op.create_table('property_private_notes',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('listing_id', sa.UUID(), nullable=False),
    sa.Column('family_member_id', sa.UUID(), nullable=False),
    sa.Column('notes', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['family_member_id'], ['lab_members.id'], ),
    sa.ForeignKeyConstraint(['listing_id'], ['property_listings.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('listing_id', 'family_member_id', name='uq_property_note_member')
    )

    op.create_table('property_reviews',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('listing_id', sa.UUID(), nullable=False),
    sa.Column('family_member_id', sa.UUID(), nullable=False),
    sa.Column('rating', sa.Integer(), nullable=False),
    sa.Column('comment', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['family_member_id'], ['lab_members.id'], ),
    sa.ForeignKeyConstraint(['listing_id'], ['property_listings.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('listing_id', 'family_member_id', name='uq_property_review_member')
    )

    op.create_table('property_share_tokens',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('listing_id', sa.UUID(), nullable=False),
    sa.Column('token_hash', sa.String(), nullable=False),
    sa.Column('created_by', sa.UUID(), nullable=False),
    sa.Column('permissions', sa.JSON(), nullable=True),
    sa.Column('expires_at', sa.DateTime(), nullable=True),
    sa.Column('max_uses', sa.Integer(), nullable=True),
    sa.Column('uses_count', sa.Integer(), nullable=True),
    sa.Column('is_revoked', sa.Boolean(), nullable=True),
    sa.Column('last_used_at', sa.DateTime(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['created_by'], ['lab_members.id'], ),
    sa.ForeignKeyConstraint(['listing_id'], ['property_listings.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('token_hash')
    )
    op.create_index('ix_property_share_tokens_listing', 'property_share_tokens', ['listing_id'], unique=False)
    op.create_index('ix_property_share_tokens_token', 'property_share_tokens', ['token_hash'], unique=False)


def downgrade() -> None:
    # Drop child tables first (FK dependencies)
    op.drop_index('ix_property_share_tokens_token', table_name='property_share_tokens')
    op.drop_index('ix_property_share_tokens_listing', table_name='property_share_tokens')
    op.drop_table('property_share_tokens')
    op.drop_table('property_reviews')
    op.drop_table('property_private_notes')
    op.drop_index('ix_property_listing_sources_listing', table_name='property_listing_sources')
    op.drop_index('ix_property_listing_sources_email', table_name='property_listing_sources')
    op.drop_table('property_listing_sources')
    op.drop_index('ix_property_due_diligence_listing', table_name='property_due_diligence')
    op.drop_table('property_due_diligence')
    op.drop_index('ix_property_documents_listing', table_name='property_documents')
    op.drop_table('property_documents')
    op.drop_table('property_collection_items')
    op.drop_index('ix_property_actions_listing', table_name='property_actions')
    op.drop_index('ix_property_actions_acted_at', table_name='property_actions')
    op.drop_table('property_actions')

    # Drop parent tables
    op.drop_index('ix_property_listings_type', table_name='property_listings')
    op.drop_index('ix_property_listings_tier', table_name='property_listings')
    op.drop_index('ix_property_listings_status', table_name='property_listings')
    op.drop_index('ix_property_listings_source', table_name='property_listings')
    op.drop_index('ix_property_listings_recommendation', table_name='property_listings')
    op.drop_index('ix_property_listings_price', table_name='property_listings')
    op.drop_index('ix_property_listings_plz', table_name='property_listings')
    op.drop_index('ix_property_listings_dedup', table_name='property_listings')
    op.drop_table('property_listings')
    op.drop_table('property_collections')
