import pytest
from unittest.mock import patch, MagicMock
from src.gemini_client import gemini

@pytest.fixture
def sample_summary():
    return [
        {"joint": 0, "mean_diff": 0.001, "std_diff": 0.002, "correlation": 0.99},
        {"joint": 1, "mean_diff": 0.001, "std_diff": 0.002, "correlation": 0.98},
    ]

def test_gemini_returns_text(sample_summary):
    prompt = "Test prompt"
    
    with patch("src.gemini_client.genai.Client") as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.models.generate_content.return_value.text = "Mock analysis text"
        
        result = gemini(sample_summary, prompt, api_key="FAKE_KEY")
        assert "Mock analysis text" in result

def test_gemini_empty_prompt(sample_summary):
    result = gemini(sample_summary, "")
    assert result is None
