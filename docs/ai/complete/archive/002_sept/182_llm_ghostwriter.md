**Context Files to Load / Create:**
* `tools/llm_ghostwriter.py` (Create)
* `requirements.txt` (Modify)

**Task: Phase 2.1 - Build the LLM Client & Caching Infrastructure**
We need a standalone module to handle LLM API calls for drafting the paper sections. To avoid unnecessary API costs, ensure deterministic rebuilds, and allow offline compilation, this module must implement a strict caching mechanism based on content hashes.

**Core Objectives:**

**1. The Ghostwriter Class (`tools/llm_ghostwriter.py`):**
* Create an `LLMGhostwriter` class.
* Use the `litellm` Python package for API calls to abstract away the specific LLM provider (this supports Gemini, OpenAI, Anthropic, and local models interchangeably without code changes).
* Implement a `draft_section(system_prompt: str, user_prompt: str, context_data: dict, model: str = "gemini/gemini-pro") -> str` method. 

**2. The Deterministic Cache:**
* We must only query the LLM if the underlying data or prompts have changed. 
* Implement caching logic inside the class (saving responses to a `paper/.cache/` directory, create if it doesn't exist). 
* The cache key should be a deterministic hash (e.g., `hashlib.sha256`) of the concatenated `system_prompt`, `user_prompt`, and `json.dumps(context_data, sort_keys=True)`. 
* If a cache file exists for that hash, load and return the text. If not, hit the API, save the response to the cache, and return it.

**3. API Key Graceful Fallback:**
* Do not hardcode API keys or specific environment variables in the code. Let `litellm` automatically handle resolving API keys from the environment (e.g., it will pick up `GEMINI_API_KEY` or `OPENAI_API_KEY` depending on the requested model).
* **Critical:** If the API key is missing, or if the API call fails (e.g., no internet, authentication error), the script MUST NOT crash. It should catch the exception, log a warning, and return a placeholder string (e.g., `"[LLM Generation Skipped: Missing API Key or Network Error]"`) so the `make paper` pipeline can still compile the LaTeX PDF.

**4. Dependencies:**
* Update `requirements.txt` to include `litellm`.

**Constraints:**
* Use standard Python `logging`. Keep logs clean (e.g., `logger.info("Cache hit for section. Skipping API call.")`).
