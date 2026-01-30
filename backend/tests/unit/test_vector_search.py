"""
Unit Tests for Vector Search

Tests embedding generation and vector similarity search.
"""
import pytest
from unittest.mock import Mock, patch
import numpy as np

from backend.core.search.embeddings import EmbeddingGenerator


class TestEmbeddingGenerator:
    """Test suite for EmbeddingGenerator"""
    
    @pytest.fixture
    def generator(self):
        """Create embedding generator"""
        return EmbeddingGenerator(model="text-embedding-3-small")
    
    @pytest.fixture
    def mock_email(self):
        """Create mock email"""
        email = Mock()
        email.id = "test-email-id"
        email.subject = "Test Subject"
        email.from_name = "Test Sender"
        email.from_address = "sender@example.com"
        email.body_markdown = "This is a test email about machine learning."
        email.body_text = None
        return email
    
    @pytest.fixture
    def mock_metadata(self):
        """Create mock metadata"""
        metadata = Mock()
        metadata.ai_category = "application-phd"
        metadata.ai_summary = "PhD application in ML"
        return metadata
    
    # =========================================================================
    # Text Preparation Tests
    # =========================================================================
    
    def test_prepare_text_with_full_data(self, generator, mock_email, mock_metadata):
        """Test text preparation with all data available"""
        text = generator._prepare_text(mock_email, mock_metadata)
        
        # Should include subject (twice for emphasis)
        assert text.count("Test Subject") == 2
        # Should include category
        assert "application-phd" in text
        # Should include sender
        assert "Test Sender" in text
        # Should include body
        assert "machine learning" in text
        # Should include summary
        assert "PhD application in ML" in text
    
    def test_prepare_text_without_metadata(self, generator, mock_email):
        """Test text preparation without metadata"""
        text = generator._prepare_text(mock_email, None)
        
        # Should still work with basic email data
        assert "Test Subject" in text
        assert "Test Sender" in text
        assert "machine learning" in text
    
    def test_prepare_text_truncates_long_body(self, generator, mock_email):
        """Test that long bodies are truncated"""
        # Create very long body
        mock_email.body_markdown = "A" * 10000
        
        text = generator._prepare_text(mock_email, None)
        
        # Should be truncated (< 6000 chars as per implementation)
        assert len(text) < 6500  # With some buffer for subject/metadata
        assert "..." in text  # Truncation indicator
    
    # =========================================================================
    # Cosine Similarity Tests
    # =========================================================================
    
    def test_cosine_similarity_identical_vectors(self, generator):
        """Test cosine similarity of identical vectors"""
        vec = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        similarity = generator.cosine_similarity(vec, vec)
        
        # Identical vectors should have similarity = 1.0
        assert abs(similarity - 1.0) < 0.001
    
    def test_cosine_similarity_orthogonal_vectors(self, generator):
        """Test cosine similarity of orthogonal vectors"""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        
        similarity = generator.cosine_similarity(vec1, vec2)
        
        # Orthogonal vectors should have similarity = 0.0
        assert abs(similarity - 0.0) < 0.001
    
    def test_cosine_similarity_opposite_vectors(self, generator):
        """Test cosine similarity of opposite vectors"""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        
        similarity = generator.cosine_similarity(vec1, vec2)
        
        # Opposite vectors get clamped to 0 (not -1) in implementation
        assert similarity >= 0.0
    
    def test_cosine_similarity_zero_vector(self, generator):
        """Test handling of zero vectors"""
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 2.0, 3.0]
        
        similarity = generator.cosine_similarity(vec1, vec2)
        
        # Should return 0.0 for zero vector
        assert similarity == 0.0
    
    # =========================================================================
    # Integration Tests (with mocked OpenAI)
    # =========================================================================
    
    @pytest.mark.asyncio
    @patch('backend.core.search.embeddings.OpenAI')
    async def test_generate_embedding_with_mock(self, mock_openai, generator, mock_email):
        """Test embedding generation with mocked OpenAI"""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_openai.return_value.embeddings.create.return_value = mock_response
        
        # Generate embedding
        embedding, content_hash = generator.generate_embedding(mock_email)
        
        # Should return 1536-dim vector
        assert len(embedding) == 1536
        # Should return content hash
        assert len(content_hash) == 64  # SHA256 hex length
    
    @pytest.mark.asyncio
    @patch('backend.core.search.embeddings.OpenAI')
    async def test_batch_embeddings_with_mock(self, mock_openai, generator):
        """Test batch embedding generation"""
        # Create mock emails
        emails = [Mock(id=f"email-{i}", subject=f"Subject {i}", 
                      body_markdown=f"Body {i}", from_name=f"Sender {i}",
                      from_address=f"sender{i}@test.com")
                 for i in range(5)]
        
        # Mock OpenAI batch response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1 * i] * 1536) for i in range(5)]
        mock_openai.return_value.embeddings.create.return_value = mock_response
        
        # Generate batch
        results = generator.generate_batch_embeddings(emails)
        
        # Should return 5 results
        assert len(results) == 5
        # Each result should be (email_id, embedding, hash)
        for email_id, embedding, content_hash in results:
            assert len(embedding) == 1536
            assert len(content_hash) == 64

