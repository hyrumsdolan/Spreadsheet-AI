import os
from dotenv import load_dotenv

# Load environment variables
# Try to load .env file but don't fail if it doesn't exist
try:
    load_dotenv()
except Exception:
    pass  # Silently continue if .env file is missing or can't be loaded

# Default API keys (can be overridden in UI)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Model options
OPENAI_MODELS = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "o4-mini"]
CLAUDE_MODELS = ["claude-3-7-sonnet-latest", "claude-3-5-haiku-latest"]