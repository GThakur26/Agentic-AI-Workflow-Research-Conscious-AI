from __future__ import annotations
import os

class API_Key_Constants:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
    XAI_API_KEY = os.getenv('XAI_API_KEY', '')

API_KEY_CONSTANTS_OBJ = API_Key_Constants()
