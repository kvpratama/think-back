from unittest.mock import MagicMock, patch

from langchain_core.prompts import ChatPromptTemplate


def test_get_prompt_pulls_from_langsmith():
    """Happy path: pull_prompt succeeds, returns the prompt."""
    mock_prompt = ChatPromptTemplate.from_messages([("system", "test prompt")])
    mock_client = MagicMock()
    mock_client.pull_prompt.return_value = mock_prompt

    with patch("src.core.prompts._get_ls_client", return_value=mock_client):
        from src.core.prompts import get_prompt

        result = get_prompt("thinkback-agent")

        assert result == mock_prompt
        mock_client.pull_prompt.assert_called_once_with("thinkback-agent:prod")


def test_get_prompt_falls_back_on_error():
    """Fallback path: pull_prompt raises, returns hardcoded default."""
    default_prompt = ChatPromptTemplate.from_messages([("system", "fallback prompt")])
    mock_client = MagicMock()
    mock_client.pull_prompt.side_effect = Exception("Network error")

    with patch("src.core.prompts._get_ls_client", return_value=mock_client):
        with patch("src.core.prompts._DEFAULTS", {"thinkback-agent": default_prompt}):
            from src.core.prompts import get_prompt

            result = get_prompt("thinkback-agent")

            assert result == default_prompt


def test_get_prompt_forwards_custom_tag():
    """Tag forwarding: get_prompt passes tag to pull_prompt."""
    mock_prompt = ChatPromptTemplate.from_messages([("system", "dev prompt")])
    mock_client = MagicMock()
    mock_client.pull_prompt.return_value = mock_prompt

    with patch("src.core.prompts._get_ls_client", return_value=mock_client):
        from src.core.prompts import get_prompt

        result = get_prompt("thinkback-agent", tag="dev")

        assert result == mock_prompt
        mock_client.pull_prompt.assert_called_once_with("thinkback-agent:dev")
