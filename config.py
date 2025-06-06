import os
import requests
import re
from functools import total_ordering
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

# --- Model Sorting ---

@total_ordering
class ReverseCmp:
    def __init__(self, obj):
        self.obj = obj
    def __eq__(self, other):
        return self.obj == other.obj
    def __lt__(self, other):
        return self.obj > other.obj

def get_model_sort_key(model_id):
    """
    Generates a sort key for model IDs.
    Sorts by version number (descending), then by name length (ascending).
    """
    version_parts = re.findall(r'(\d+(?:\.\d+)?)', model_id)
    if not version_parts:
        numeric_parts = (0,)
    else:
        numeric_parts = tuple(float(p) for p in version_parts)
    
    length = len(model_id)
    return (ReverseCmp(numeric_parts), length)

# --- Model Fetching ---

def get_openai_models(api_key):
    """Fetches and sorts available OpenAI models from the API."""
    default_models = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "o4-mini"]
    if not api_key:
        return default_models, None
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get("https://api.openai.com/v1/models", headers=headers)
        response.raise_for_status()
        models = response.json().get("data", [])

        # Define patterns to exclude
        exclude_keywords = ["audio", "realtime", "transcribe", "computer", "vision", "image", "tts", "search", "3.5"]
        four_digits_pattern = re.compile(r'\d{4}')
        
        # Filter for relevant chat models
        chat_models = [
            m for m in models 
            if (("gpt" in m["id"] or "o4" in m["id"]) and
                not any(keyword in m["id"] for keyword in exclude_keywords) and
                not four_digits_pattern.search(m["id"]))
        ]
        
        model_ids = [m["id"] for m in chat_models]
        model_list = sorted(model_ids, key=get_model_sort_key)
        
        return model_list, None
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            return default_models, "Invalid OpenAI API key."
        return default_models, f"Error fetching OpenAI models: {e}"
    except (requests.exceptions.RequestException, KeyError) as e:
        return default_models, f"Error fetching OpenAI models: {e}"

def get_claude_models(api_key):
    """Fetches available Claude models from the API."""
    default_models = ["claude-3-7-sonnet-latest", "claude-3-5-haiku-latest"]
    if not api_key:
        return default_models, None
    try:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
        response = requests.get("https://api.anthropic.com/v1/models", headers=headers)
        response.raise_for_status()
        models = response.json().get("data", [])
        
        model_ids = [m["id"] for m in models if "claude" in m["id"]]
        
        model_list = sorted(model_ids, key=get_model_sort_key)
        return model_list, None
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            return default_models, "Invalid Claude API key."
        return default_models, f"Error fetching Claude models: {e}"
    except (requests.exceptions.RequestException, KeyError) as e:
        return default_models, f"Error fetching Claude models: {e}"

# Model options - Fallback defaults
OPENAI_MODELS = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "o4-mini"]
CLAUDE_MODELS = ["claude-3-7-sonnet-latest", "claude-3-5-haiku-latest"]