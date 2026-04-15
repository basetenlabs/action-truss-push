"""Unit tests for src/main.py."""

import os
import sys
from unittest import mock

import pytest
import requests

# Mock the truss SDK before importing main — it's not installed locally.
sys.modules["truss"] = mock.MagicMock()
sys.modules["truss_chains"] = mock.MagicMock()
sys.modules["truss_chains.framework"] = mock.MagicMock()
sys.modules["truss_chains.deployment"] = mock.MagicMock()
sys.modules["truss_chains.deployment.deployment_client"] = mock.MagicMock()
sys.modules["truss_chains.private_types"] = mock.MagicMock()

from src import main


# ---------------------------------------------------------------------------
# main — config validation
# ---------------------------------------------------------------------------


class TestMainValidation:
    def test_regional_without_environment_exits(self):
        env = {
            "TRUSS_DIRECTORY": ".github/tests/model",
            "BASETEN_API_KEY": "key",
            "REGIONAL_ENVIRONMENT": "true",
            "ENVIRONMENT": "",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                main.main()
            assert "regional-environment" in str(exc_info.value)
            assert "no environment" in str(exc_info.value)


# ---------------------------------------------------------------------------
# build_deployment_name
# ---------------------------------------------------------------------------


class TestBuildDeploymentName:
    def test_pr_ref(self):
        env = {"GITHUB_SHA": "abc1234def5678", "GITHUB_REF": "refs/pull/42/merge"}
        with mock.patch.dict(os.environ, env):
            assert main.build_deployment_name() == "PR-42_abc1234"

    def test_non_pr_ref(self):
        env = {"GITHUB_SHA": "abc1234def5678", "GITHUB_REF": "refs/heads/main"}
        with mock.patch.dict(os.environ, env):
            assert main.build_deployment_name() == "abc1234"

    def test_missing_sha(self):
        env = {"GITHUB_REF": "refs/heads/main"}
        with mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop("GITHUB_SHA", None)
            result = main.build_deployment_name()
            assert result == "unknown"

    def test_sha_truncated_to_7(self):
        env = {"GITHUB_SHA": "a" * 40, "GITHUB_REF": "refs/heads/main"}
        with mock.patch.dict(os.environ, env):
            assert main.build_deployment_name() == "a" * 7


# ---------------------------------------------------------------------------
# get_predict_payload
# ---------------------------------------------------------------------------


class TestGetPredictPayload:
    def test_override_takes_precedence(self):
        config = {"model_metadata": {"example_model_input": {"text": "default"}}}
        result = main.get_predict_payload(config, '{"text": "override"}')
        assert result == {"text": "override"}

    def test_falls_back_to_example_model_input(self):
        config = {"model_metadata": {"example_model_input": {"text": "example"}}}
        result = main.get_predict_payload(config, "")
        assert result == {"text": "example"}

    def test_returns_none_when_no_payload(self):
        config = {"model_metadata": {}}
        result = main.get_predict_payload(config, "")
        assert result is None

    def test_returns_none_when_no_metadata(self):
        config = {}
        result = main.get_predict_payload(config, "")
        assert result is None


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_loads_test_model_config(self):
        config = main.load_config(".github/tests/model")
        assert "model_name" in config
        assert config["model_name"] == "truss-push-action test-model"


# ---------------------------------------------------------------------------
# predict — URL construction
# ---------------------------------------------------------------------------


class TestPredictUrl:
    """Verify that predict() constructs the correct URL based on regional flag."""

    @mock.patch("src.main.requests.post")
    def test_default_uses_deployment_endpoint(self, mock_post):
        mock_resp = mock.Mock()
        mock_resp.text = '{"result": "ok"}'
        mock_resp.raise_for_status = mock.Mock()
        mock_post.return_value = mock_resp

        main.predict("model123", "dep456", "key", {"input": "x"}, 30)

        called_url = mock_post.call_args[0][0]
        assert called_url == (
            "https://model-model123.api.baseten.co/deployment/dep456/predict"
        )

    @mock.patch("src.main.requests.post")
    def test_regional_with_environment(self, mock_post):
        mock_resp = mock.Mock()
        mock_resp.text = '{"result": "ok"}'
        mock_resp.raise_for_status = mock.Mock()
        mock_post.return_value = mock_resp

        main.predict(
            "model123", "dep456", "key", {"input": "x"}, 30,
            environment="prod-us", regional=True,
        )

        called_url = mock_post.call_args[0][0]
        assert called_url == (
            "https://model-model123-prod-us.api.baseten.co/predict"
        )

    @mock.patch("src.main.requests.post")
    def test_regional_without_environment_falls_back(self, mock_post):
        mock_resp = mock.Mock()
        mock_resp.text = '{"result": "ok"}'
        mock_resp.raise_for_status = mock.Mock()
        mock_post.return_value = mock_resp

        main.predict(
            "model123", "dep456", "key", {"input": "x"}, 30,
            environment=None, regional=True,
        )

        called_url = mock_post.call_args[0][0]
        assert called_url == (
            "https://model-model123.api.baseten.co/deployment/dep456/predict"
        )

    @mock.patch("src.main.requests.post")
    def test_non_regional_with_environment_uses_deployment(self, mock_post):
        mock_resp = mock.Mock()
        mock_resp.text = '{"result": "ok"}'
        mock_resp.raise_for_status = mock.Mock()
        mock_post.return_value = mock_resp

        main.predict(
            "model123", "dep456", "key", {"input": "x"}, 30,
            environment="prod-us", regional=False,
        )

        called_url = mock_post.call_args[0][0]
        assert called_url == (
            "https://model-model123.api.baseten.co/deployment/dep456/predict"
        )

    @mock.patch("src.main.requests.post")
    def test_passes_auth_header(self, mock_post):
        mock_resp = mock.Mock()
        mock_resp.text = '{"result": "ok"}'
        mock_resp.raise_for_status = mock.Mock()
        mock_post.return_value = mock_resp

        main.predict("m", "d", "my-api-key", {"input": "x"}, 30)

        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Api-Key my-api-key"

    @mock.patch("src.main.requests.post")
    def test_passes_payload_as_json(self, mock_post):
        mock_resp = mock.Mock()
        mock_resp.text = '{"result": "ok"}'
        mock_resp.raise_for_status = mock.Mock()
        mock_post.return_value = mock_resp

        payload = {"text": "hello", "max_tokens": 100}
        main.predict("m", "d", "key", payload, 30)

        assert mock_post.call_args[1]["json"] == payload


# ---------------------------------------------------------------------------
# predict_chain — URL construction
# ---------------------------------------------------------------------------


class TestPredictChainUrl:
    """Verify that predict_chain() constructs the correct URL."""

    @mock.patch("src.main.requests.post")
    def test_default_uses_deployment_endpoint(self, mock_post):
        mock_resp = mock.Mock()
        mock_resp.text = '{"result": "ok"}'
        mock_resp.raise_for_status = mock.Mock()
        mock_post.return_value = mock_resp

        main.predict_chain("chain123", "dep456", "key", {"input": "x"}, 30)

        called_url = mock_post.call_args[0][0]
        assert called_url == (
            "https://chain-chain123.api.baseten.co"
            "/deployment/dep456/run_remote"
        )

    @mock.patch("src.main.requests.post")
    def test_regional_with_environment(self, mock_post):
        mock_resp = mock.Mock()
        mock_resp.text = '{"result": "ok"}'
        mock_resp.raise_for_status = mock.Mock()
        mock_post.return_value = mock_resp

        main.predict_chain(
            "chain123", "dep456", "key", {"input": "x"}, 30,
            environment="dev-us", regional=True,
        )

        called_url = mock_post.call_args[0][0]
        assert called_url == (
            "https://chain-chain123-dev-us.api.baseten.co/run_remote"
        )

    @mock.patch("src.main.requests.post")
    def test_regional_without_environment_falls_back(self, mock_post):
        mock_resp = mock.Mock()
        mock_resp.text = '{"result": "ok"}'
        mock_resp.raise_for_status = mock.Mock()
        mock_post.return_value = mock_resp

        main.predict_chain(
            "chain123", "dep456", "key", {"input": "x"}, 30,
            environment=None, regional=True,
        )

        called_url = mock_post.call_args[0][0]
        assert called_url == (
            "https://chain-chain123.api.baseten.co"
            "/deployment/dep456/run_remote"
        )

    @mock.patch("src.main.requests.post")
    def test_non_regional_with_environment_uses_deployment(self, mock_post):
        mock_resp = mock.Mock()
        mock_resp.text = '{"result": "ok"}'
        mock_resp.raise_for_status = mock.Mock()
        mock_post.return_value = mock_resp

        main.predict_chain(
            "chain123", "dep456", "key", {"input": "x"}, 30,
            environment="dev-us", regional=False,
        )

        called_url = mock_post.call_args[0][0]
        assert called_url == (
            "https://chain-chain123.api.baseten.co"
            "/deployment/dep456/run_remote"
        )


# ---------------------------------------------------------------------------
# predict — response handling
# ---------------------------------------------------------------------------


class TestPredictResponse:
    @mock.patch("src.main.requests.post")
    def test_sync_returns_expected_fields(self, mock_post):
        mock_resp = mock.Mock()
        mock_resp.text = '{"output": "hello"}'
        mock_resp.raise_for_status = mock.Mock()
        mock_post.return_value = mock_resp

        result = main.predict("m", "d", "key", {"input": "x"}, 30)

        assert result["response"] == '{"output": "hello"}'
        assert result["streaming"] is False
        assert result["tokens"] == 0
        assert "total_time" in result
        assert "ttfb" in result

    @mock.patch("src.main.requests.post")
    def test_sync_truncates_response_to_4096(self, mock_post):
        mock_resp = mock.Mock()
        mock_resp.text = "x" * 8000
        mock_resp.raise_for_status = mock.Mock()
        mock_post.return_value = mock_resp

        result = main.predict("m", "d", "key", {"input": "x"}, 30)

        assert len(result["response"]) == 4096

    @mock.patch("src.main.requests.post")
    def test_raises_on_http_error(self, mock_post):
        mock_resp = mock.Mock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("403 Forbidden")
        mock_post.return_value = mock_resp

        with pytest.raises(requests.HTTPError):
            main.predict("m", "d", "key", {"input": "x"}, 30)


# ---------------------------------------------------------------------------
# _predict_streaming — SSE parsing
# ---------------------------------------------------------------------------


class TestPredictStreaming:
    @mock.patch("src.main.requests.post")
    def test_parses_sse_chunks(self, mock_post):
        sse_lines = [
            'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            'data: {"choices": [{"delta": {"content": " world"}}]}',
            "data: [DONE]",
        ]
        mock_resp = mock.Mock()
        mock_resp.raise_for_status = mock.Mock()
        mock_resp.iter_lines.return_value = iter(sse_lines)
        mock_post.return_value = mock_resp

        result = main.predict("m", "d", "key", {"stream": True}, 30)

        assert result["streaming"] is True
        assert result["response"] == "Hello world"
        assert result["tokens"] == 2

    @mock.patch("src.main.requests.post")
    def test_skips_empty_and_non_data_lines(self, mock_post):
        sse_lines = [
            "",
            ": comment",
            'data: {"choices": [{"delta": {"content": "ok"}}]}',
            "",
            "data: [DONE]",
        ]
        mock_resp = mock.Mock()
        mock_resp.raise_for_status = mock.Mock()
        mock_resp.iter_lines.return_value = iter(sse_lines)
        mock_post.return_value = mock_resp

        result = main.predict("m", "d", "key", {"stream": True}, 30)

        assert result["response"] == "ok"
        assert result["tokens"] == 1

    @mock.patch("src.main.requests.post")
    def test_handles_malformed_json(self, mock_post):
        sse_lines = [
            "data: {not valid json}",
            'data: {"choices": [{"delta": {"content": "ok"}}]}',
            "data: [DONE]",
        ]
        mock_resp = mock.Mock()
        mock_resp.raise_for_status = mock.Mock()
        mock_resp.iter_lines.return_value = iter(sse_lines)
        mock_post.return_value = mock_resp

        result = main.predict("m", "d", "key", {"stream": True}, 30)

        assert result["response"] == "ok"
        assert result["tokens"] == 1
