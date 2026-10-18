from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

# Define as instruções principais do assistente jurídico
system_block = """Você é um assistente jurídico que responde usando estritamente o conteúdo do PDF fornecido quando possível.
Se a resposta não estiver no PDF, diga que não encontrou no documento e ofereça passos de verificação.
Formate a resposta com: Resumo, Fundamentação (com citações de trechos entre aspas) e Próximos passos.
Se houver conflito entre o PDF e conhecimento externo, priorize o PDF e sinalize a divergência."""

def create_qa_prompt() -> ChatPromptTemplate:
    """
    Cria o template de prompt que será usado para formatar perguntas e contexto do PDF.
    """
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            # Define o papel de sistema com as instruções base
            SystemMessage(content = system_block),
            # Define a estrutura da mensagem humana com pergunta e contexto
            (
                "human", 
                "Pergunta: {question}\n\n"
                "Contexto do PDF:\n{context}\n\n"
                "Responda de forma sucinta, técnica e didática."
               ),
        ]
    )    
    
    return qa_prompt
