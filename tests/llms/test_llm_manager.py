import pytest
from unittest.mock import MagicMock, patch

from app.llms.llm_manager import LLMManager
from app.core.exceptions.llm_exceptions import (
    LLMRateLimitError,
    LLMTimeoutError,
    LLMServiceUnavailableError,
    LLMGenerationError,
    LLMUnexpectedError,
    LLMManagerError,
)


class TestLLMManager:
    @pytest.fixture(autouse=True)
    def setup_llm_manager(self):
        self.llm_manager = LLMManager()
        self.llm_manager.settings = MagicMock()
        self.llm_manager.settings.model = ["groq-model-1", "groq-model-2"]

    @patch("time.sleep", return_value=None)
    def test_success_on_first_try(self, _):
        self.llm_manager.generate_response = MagicMock(return_value="ok")
        result = self.llm_manager.call_llm_with_retry(prompt="test")

        assert result == "ok"
        self.llm_manager.generate_response.assert_called_once()

    @patch("time.sleep", return_value=None)
    def test_rate_limit_then_success(self, _):
        self.llm_manager.generate_response = MagicMock(
            side_effect=[LLMRateLimitError(), "success"]
        )

        result = self.llm_manager.call_llm_with_retry(prompt="test")

        assert result == "success"
        assert self.llm_manager.generate_response.call_count == 2

    @patch("time.sleep", return_value=None)
    def test_timeout_then_success(self, _):
        self.llm_manager.generate_response = MagicMock(
            side_effect=[LLMTimeoutError(), "success"]
        )
        result = self.llm_manager.call_llm_with_retry(prompt="test")
        assert result == "success"
        assert self.llm_manager.generate_response.call_count == 2

    @patch("time.sleep", return_value=None)
    def test_service_unavailable_then_success(self, _):
        self.llm_manager.generate_response = MagicMock(
            side_effect=[LLMServiceUnavailableError(), "success"]
        )
        result = self.llm_manager.call_llm_with_retry(prompt="test")
        assert result == "success"
        assert self.llm_manager.generate_response.call_count == 2

    @patch("time.sleep", return_value=None)
    def test_unrecoverable_generation_error(self, _):
        self.llm_manager.generate_response = MagicMock(
            side_effect=LLMGenerationError("fatal")
        )
        with pytest.raises(LLMManagerError):
            self.llm_manager.call_llm_with_retry(prompt="test")
        assert self.llm_manager.generate_response.call_count == 2
        assert "fatal" in str(self.llm_manager.generate_response.side_effect)

    @patch("time.sleep", return_value=None)
    def test_unrecoverable_unexpected_error(self, _):
        self.llm_manager.generate_response = MagicMock(
            side_effect=LLMUnexpectedError("unexpected")
        )
        with pytest.raises(LLMManagerError):
            self.llm_manager.call_llm_with_retry(prompt="test")
        assert self.llm_manager.generate_response.call_count == 2
        assert "unexpected" in str(self.llm_manager.generate_response.side_effect)

    @patch("time.sleep", return_value=None)
    def test_all_models_fail_due_to_service_unavailable(self, _):
        self.llm_manager.generate_response = MagicMock(
            side_effect=LLMServiceUnavailableError()
        )

        with pytest.raises(LLMManagerError):
            self.llm_manager.call_llm_with_retry(max_retries=2, prompt="test")
        assert self.llm_manager.generate_response.call_count == 4
