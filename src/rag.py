
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    
    out = []
    
    # Percorre os documentos e extrai conteúdo e metadados (como número da página)
    for d in docs:
        meta = d.metadata or {}
        where = f"p.{meta.get('page', '?')}"
        out.append(f'[{where}] "{d.page_content.strip()[:800]}{"…" if len(d.page_content) > 800 else ""}"')
    
    # Junta todos os trechos formatados em um único texto
    return "\n\n".join(out)

def create_rag_pipeline(retriever, prompt, llm):
    return (
        RunnableParallel(
            context=retriever | format_docs,
            question=RunnablePassthrough(),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
# Explicação do rag_pipeline:

# Em nosso fluxo, a conversão da pergunta em vetor acontece dentro do ramo context = retriever | format_docs, exatamente no momento 
# em que o retriever é executado quando você chama rag_pipeline.invoke(pergunta). O retriever criado a partir do Chroma chama internamente 
# o vectordb.similarity_search(pergunta, k=3), que por sua vez executa embeddings.embed_query(pergunta) usando o modelo 
# que você definiu (sentence-transformers/msmarco-bert-base-dot-v5). Esse vetor da query é então comparado aos vetores dos chunks já indexados 
# (gerados antes em Chroma.from_documents(...)). Em paralelo, o RunnablePassthrough() só carrega a string da pergunta adiante, sem vetorização. 
# Ou seja, a query do usuário é vetorizada toda vez que o retriever roda, antes da busca de similaridade no Chroma, 
# enquanto os vetores dos documentos já estavam persistidos desde a etapa de indexação.

# O RunnableParallel dispara em paralelo o ramo que busca e formata o contexto a partir do retriever e o ramo que apenas preserva a pergunta. 
# Isso é útil porque o ChatPromptTemplate espera receber um objeto com campos nomeados, por exemplo {context, question}, e você ganha eficiência 
# por não fazer esses passos em série quando não há dependência entre eles.
