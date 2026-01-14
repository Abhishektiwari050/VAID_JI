"""
Test suite for VAID JI Medical Research Assistant
Run with: pytest tests/
"""

import pytest
import os
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_imports():
    """Test that all modules can be imported"""
    try:
        import config
        import pdf_preprocessing
        import rag_pipeline
        import auto_summary
        import rephrasing_loop
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import module: {e}")


def test_config_validation():
    """Test configuration validation"""
    from config import validate_config, get_config_summary
    
    # Get config summary should not raise
    summary = get_config_summary()
    assert isinstance(summary, dict)
    assert 'embedding_model' in summary
    assert 'temperature' in summary


def test_chunk_text():
    """Test text chunking functionality"""
    from pdf_preprocessing import chunk_text
    
    # Test with sample text
    text = "This is a test. " * 100  # Create text longer than chunk size
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)
    

def test_chunk_text_empty():
    """Test chunking with empty text"""
    from pdf_preprocessing import chunk_text
    
    chunks = chunk_text("")
    assert chunks == []


def test_get_cache_key():
    """Test cache key generation"""
    from rag_pipeline import get_cache_key
    
    key1 = get_cache_key("test query")
    key2 = get_cache_key("test query")
    key3 = get_cache_key("different query")
    
    # Same input should produce same hash
    assert key1 == key2
    # Different input should produce different hash
    assert key1 != key3
    # Hash should be hex string
    assert all(c in '0123456789abcdef' for c in key1)


def test_is_response_low_quality():
    """Test response quality checker"""
    from rephrasing_loop import is_response_low_quality
    
    # Good response - long with medical terms
    good_response = "The clinical study examined treatment protocols for 150 patients with positive diagnosis results."
    assert not is_response_low_quality(good_response)
    
    # Bad response - too short
    bad_response = "Yes."
    assert is_response_low_quality(bad_response)
    
    # Bad response - no medical terms
    non_medical = "This is a long response but it contains no medical terminology at all whatsoever."
    assert is_response_low_quality(non_medical)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
