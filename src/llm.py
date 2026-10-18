from langchain_groq import ChatGroq
from config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

def create_llm(api_key: str) -> ChatGroq:
    """
    Cria e retorna uma instância do modelo de linguagem ChatGroq com base na chave de API fornecida.
    """
    
    return ChatGroq(
        api_key=api_key,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )