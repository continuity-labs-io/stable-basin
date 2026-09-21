import logging
from google import genai

logger = logging.getLogger(__name__)

def get_client() -> genai.Client:
    return genai.Client()

def get_best_model(client: genai.Client, base_name: str = "gemini") -> str:
    """
    Finds the best available model for the given base_name.
    Returns it in a litellm compatible format (e.g. gemini/gemini-2.5-pro).
    """
    try:
        models = list(client.models.list())
        model_names = [m.name.replace('models/', '') for m in models if m.name.startswith(f'models/{base_name}')]
        
        if not model_names:
            return f"gemini/{base_name}-2.5-pro"
            
        # Prefer pro models, non-preview if possible
        pro_models = [m for m in model_names if 'pro' in m and 'preview' not in m]
        if pro_models:
            # Sort to pick highest version
            best = sorted(pro_models, reverse=True)[0]
            logger.info(f"Selected best stable pro model: {best}")
            return f"gemini/{best}"
            
        pro_preview = [m for m in model_names if 'pro' in m]
        if pro_preview:
            best = sorted(pro_preview, reverse=True)[0]
            logger.info(f"Selected best preview pro model: {best}")
            return f"gemini/{best}"
            
        best = sorted(model_names, reverse=True)[0]
        logger.info(f"Selected fallback model: {best}")
        return f"gemini/{best}"
            
    except Exception as e:
        logger.warning(f"Failed to fetch models from API: {e}")
        
    return f"gemini/{base_name}-2.5-pro" # safe fallback
