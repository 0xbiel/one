from app.config import Settings
from app.integrations import LMStudioAdapter


def test_openai_compatible_overrides_preserve_lm_studio_defaults():
    defaults = Settings(_env_file=None)
    assert defaults.effective_llm_base_url == "http://127.0.0.1:1234/v1"
    assert defaults.effective_llm_model == "qwen3.6-35b-a3b"

    settings = Settings(
        _env_file=None,
        llm_provider="openrouter",
        llm_base_url="https://openrouter.ai/api/v1",
        llm_model="provider/model",
        llm_api_key="secret-value",
        llm_timeout_seconds=4.5,
    )
    assert settings.effective_llm_base_url == "https://openrouter.ai/api/v1"
    assert settings.effective_llm_model == "provider/model"
    assert settings.effective_llm_api_key == "secret-value"
    assert settings.llm_provider == "openrouter"


def test_disabled_llm_never_attempts_request():
    settings = Settings(_env_file=None, llm_enabled=False)
    adapter = LMStudioAdapter(settings)
    assert adapter.summarize({"sample": True}) is None
    assert adapter.last_error == "disabled"
    assert adapter.family_summary({"sample": True}) is None
    assert adapter.last_error == "disabled"
