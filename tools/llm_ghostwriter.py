import os
import json
import hashlib
import logging
from pathlib import Path
import litellm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class LLMGhostwriter:
    """
    Standalone module to handle LLM API calls for drafting paper sections with a deterministic cache.
    """
    def __init__(self, cache_dir: str = "paper/.cache"):
        self.root_dir = Path(__file__).resolve().parent.parent
        self.cache_dir = self.root_dir / cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _generate_cache_key(self, system_prompt: str, user_prompt: str, context_data: dict) -> str:
        """
        Generate a deterministic hash based on the prompts and context data.
        """
        data_str = json.dumps(context_data, sort_keys=True)
        raw_key = system_prompt + user_prompt + data_str
        return hashlib.sha256(raw_key.encode('utf-8')).hexdigest()

    def draft_section(self, system_prompt: str, user_prompt: str, context_data: dict, model: str = None) -> str:
        """
        Drafts a section using the specified LLM. Checks the cache first to avoid unnecessary API calls.
        """
        if model is None:
            try:
                from genai_client import get_client, get_best_model
                client = get_client()
                model = get_best_model(client)
            except Exception as e:
                logger.warning(f"Failed to auto-select model: {e}")
                model = "gemini/gemini-2.5-pro"
                
        cache_key = self._generate_cache_key(system_prompt, user_prompt, context_data)
        cache_file = self.cache_dir / f"{cache_key}.txt"

        if cache_file.exists():
            logger.info("Cache hit for section. Skipping API call.")
            with open(cache_file, "r", encoding="utf-8") as f:
                return f.read()

        logger.info(f"Cache miss. Querying LLM (model: {model})...")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{user_prompt}\n\nContext Data:\n{json.dumps(context_data, indent=2)}"}
        ]

        try:
            # litellm handles API keys automatically from environment variables
            # (e.g., GEMINI_API_KEY, OPENAI_API_KEY, etc.)
            response = litellm.completion(
                model=model,
                messages=messages
            )
            generated_text = response.choices[0].message.content

            # Save to cache
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(generated_text)
            
            return generated_text

        except Exception as e:
            logger.warning(f"LLM API call failed: {e}")
            return "[LLM Generation Skipped: Missing API Key or Network Error]"

