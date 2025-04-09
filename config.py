"""
Configuration defaults for the Translator application.
This file contains the default values for environment variables.
These are only used if the actual environment variables are not set.
"""

# Default model configuration
DEFAULT_CONFIG = {
    "GEMMA_GPU_LAYERS": 48,
    "GEMMA_CONTEXT_SIZE": 8192,
    "MAX_PARALLEL_MODELS": 4,
    "SESSION_TIMEOUT": 1800,
    "MODEL_INSTANCE_TIMEOUT": 1800,
    "ALLOW_GPU": True,
    "LOGLEVEL": "INFO",
    "MAX_TOKENS": 4096,
    "CHUNK_SIZE": 3000
}

# Convert boolean and integer values to strings for environment variables
ENV_DEFAULTS = {
    key: str(value).lower() if isinstance(value, bool) else str(value)
    for key, value in DEFAULT_CONFIG.items()
} 