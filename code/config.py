
import os
import logging
from dotenv import load_dotenv
from typing import Optional, Dict, Any

# Load environment variables from .env file if present
# (This is safe for local development; in production, env vars should be set by the environment)
load_dotenv()

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class AgentConfig:
    """
    Configuration management for Custom Translator Application Agent.
    Handles environment variable loading, API key management, LLM config,
    domain-specific settings, validation, error handling, and defaults.
    """

    # Required environment variables for all core services
    REQUIRED_KEYS = [
        "AZURE_BLOB_STORAGE_CONNECTION_STRING",
        "AZURE_BLOB_CONTAINER_NAME",
        "AZURE_TRANSLATOR_API_KEY",
        "AZURE_TRANSLATOR_ENDPOINT",
        "AZURE_TRANSLATOR_REGION",
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_API_KEY",
        "AZURE_SEARCH_INDEX_NAME",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
    ]

    # LLM configuration defaults
    LLM_CONFIG_DEFAULTS = {
        "provider": "openai",
        "model": "gpt-4.1",
        "temperature": 0.7,
        "max_tokens": 2000,
        "system_prompt": (
            "You are a formal, reliable assistant for a custom translator application. Your primary role is to:\n\n"
            "- Accept a filename as input.\n\n"
            "- Retrieve the file from Azure Blob Storage using the configured blob settings.\n\n"
            "- Generate a secure SAS URL for the file.\n\n"
            "- Submit the SAS URL to the Azure Translator service.\n\n"
            "- If the translation service returns a status code of 200, proceed to provide the translated content.\n\n"
            "- If the status is not 200, retry the translation request for up to 100 seconds before returning an error.\n\n"
            "- Additionally, answer user queries using the connected knowledge base via Azure AI Search, referencing only the provided source documents.\n\n"
            "Output all responses in clear, formal language. If information is not found in the knowledge base, politely inform the user and suggest alternative actions."
        ),
        "user_prompt_template": "Please provide the filename you wish to translate, or ask a question related to the knowledge base.",
        "few_shot_examples": [
            "Translate the file report_2023.docx.",
            "What is the primary function of leaves in plants?"
        ]
    }

    # Domain-specific settings
    DOMAIN_SETTINGS = {
        "blob_sas_expiry_minutes": 10,
        "translation_retry_seconds": 100,
        "rag": {
            "enabled": True,
            "retrieval_service": "azure_ai_search",
            "embedding_model": "text-embedding-ada-002",
            "top_k": 5,
            "search_type": "vector_semantic"
        },
        "output_format": {
            "translation": "Provide a summary of the translation process and the translated content.",
            "knowledge_base": "Provide concise, referenced answers using information from the listed source documents.",
            "error": "Return a clear, formal error message."
        },
        "fallback_response": "I am unable to find the requested information in the knowledge base. Please provide more details or consult an alternative resource."
    }

    def __init__(self):
        self._env = os.environ
        self._logger = logging.getLogger("AgentConfig")
        self._validate_env()

    def _validate_env(self):
        """Validate that all required environment variables are present."""
        missing = [k for k in self.REQUIRED_KEYS if not self._env.get(k)]
        if missing:
            self._logger.error(f"Missing required environment variables: {missing}")
            raise ConfigError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

    def get(self, key: str, default: Optional[Any] = None) -> Optional[str]:
        """Get an environment variable with optional default."""
        return self._env.get(key, default)

    def get_llm_config(self) -> Dict[str, Any]:
        """Return LLM configuration, using defaults and allowing overrides via env vars."""
        config = self.LLM_CONFIG_DEFAULTS.copy()
        config["provider"] = self.get("LLM_PROVIDER", config["provider"])
        config["model"] = self.get("LLM_MODEL", config["model"])
        config["temperature"] = float(self.get("LLM_TEMPERATURE", config["temperature"]))
        config["max_tokens"] = int(self.get("LLM_MAX_TOKENS", config["max_tokens"]))
        config["system_prompt"] = self.get("LLM_SYSTEM_PROMPT", config["system_prompt"])
        config["user_prompt_template"] = self.get("LLM_USER_PROMPT_TEMPLATE", config["user_prompt_template"])
        # few_shot_examples can be overridden by a comma-separated env var
        examples = self.get("LLM_FEW_SHOT_EXAMPLES")
        if examples:
            config["few_shot_examples"] = [e.strip() for e in examples.split(",") if e.strip()]
        return config

    def get_blob_config(self) -> Dict[str, str]:
        """Return Azure Blob Storage configuration."""
        return {
            "connection_string": self.get("AZURE_BLOB_STORAGE_CONNECTION_STRING"),
            "container_name": self.get("AZURE_BLOB_CONTAINER_NAME")
        }

    def get_translator_config(self) -> Dict[str, str]:
        """Return Azure Translator configuration."""
        return {
            "api_key": self.get("AZURE_TRANSLATOR_API_KEY"),
            "endpoint": self.get("AZURE_TRANSLATOR_ENDPOINT"),
            "region": self.get("AZURE_TRANSLATOR_REGION"),
            "target_language": self.get("AZURE_TRANSLATOR_TARGET_LANGUAGE", "en")
        }

    def get_search_config(self) -> Dict[str, str]:
        """Return Azure AI Search configuration."""
        return {
            "endpoint": self.get("AZURE_SEARCH_ENDPOINT"),
            "api_key": self.get("AZURE_SEARCH_API_KEY"),
            "index_name": self.get("AZURE_SEARCH_INDEX_NAME")
        }

    def get_openai_config(self) -> Dict[str, str]:
        """Return Azure OpenAI configuration."""
        return {
            "endpoint": self.get("AZURE_OPENAI_ENDPOINT"),
            "api_key": self.get("AZURE_OPENAI_API_KEY"),
            "embedding_deployment": self.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            "deployment": self.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4-1106-preview")
        }

    def get_domain_settings(self) -> Dict[str, Any]:
        """Return domain-specific settings."""
        return self.DOMAIN_SETTINGS.copy()

    def get_fallback_response(self) -> str:
        """Return fallback response for knowledge base misses."""
        return self.DOMAIN_SETTINGS["fallback_response"]

    def get_output_format(self) -> Dict[str, str]:
        """Return output formatting instructions."""
        return self.DOMAIN_SETTINGS["output_format"]

    def as_dict(self) -> Dict[str, Any]:
        """Return all configuration as a dictionary."""
        return {
            "blob": self.get_blob_config(),
            "translator": self.get_translator_config(),
            "search": self.get_search_config(),
            "openai": self.get_openai_config(),
            "llm": self.get_llm_config(),
            "domain": self.get_domain_settings()
        }

# Example usage:
# try:
#     config = AgentConfig()
#     llm_cfg = config.get_llm_config()
# except ConfigError as e:
#     print(f"Configuration error: {e}")

