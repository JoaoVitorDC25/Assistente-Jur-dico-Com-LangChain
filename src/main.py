import os
import tempfile
import streamlit as st

from llm import create_llm
from indexacao import create_vetorial_db
from prompts import create_qa_prompt
from rag import create_rag_pipeline


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    st.set_page_config(page_title = "Data Science Academy", page_icon = ":100:", layout = "wide")

    with st.sidebar:
        st.header("Configurações")
        api_key = st.text_input("Coloque aqui sua GROQ API Key e pressione Enter", type = "password")
        st.divider()
        st.subheader("Instruções")
        st.write("1) Informe sua chave no campo acima.\n2) Digite sua pergunta ou dúvida.\n3) Clique em Enviar.")
        st.info("Aviso: a IA pode cometer erros. Verifique fatos críticos.")

    st.title("⚖️ Assistente Jurídico (Com o Uso de RAG)")
    st.caption("Modelo: openai/gpt-oss-20b via Groq + LangChain")

    if not api_key:
        st.warning("Informe a GROQ API Key na barra lateral para continuar.")
        st.stop()

    os.environ["GROQ_API_KEY"] = api_key

    llm = create_llm(api_key)

    pdf_file = st.file_uploader("Envie um PDF da área jurídica (contrato, parecer, decisão, lei consolidada…)", type = ["pdf"])

    # Cria um diretório temporário para armazenar os vetores do ChromaDB
    if "chroma_dir" not in st.session_state:
        st.session_state.chroma_dir = tempfile.mkdtemp(prefix = "chroma_rag_")

    if pdf_file and "vectordb_ready" not in st.session_state:
        with st.spinner("Indexando o PDF no ChromaDB…"):
            st.session_state.vectordb = create_vetorial_db(pdf_file.read())
            st.session_state.vectordb_ready = True
            st.success("Indexação concluída.")

    retriever = None

    if st.session_state.get("vectordb_ready"):
        retriever = st.session_state.vectordb.as_retriever(search_kwargs = {"k": 3})

    # Nesse trecho acima, a estratégia matemática é busca vetorial por similaridade, mais especificamente, 
    # uma busca por vizinhos mais próximos (k-NN, “k-nearest neighbors”) baseada em distância de similaridade coseno.

    # Quando você chama vectordb.as_retriever(search_kwargs={"k": 3}), o Chroma é instruído a, para cada nova consulta (query), 
    # converter a pergunta em vetor de embeddings e depois calcular a proximidade angular entre esse vetor e os vetores armazenados dos chunks do PDF. 
    # Essa proximidade é medida pelo cosseno do ângulo entre os vetores: quanto menor o ângulo (maior o cosseno, mais próximo de 1), 
    # maior a similaridade semântica.

    # Assim, o retriever retorna os 3 vetores mais próximos (os 3 blocos de texto mais semanticamente relacionados à pergunta). Ou seja, 
    # a base matemática é: similaridade_coseno(A, B) = (A · B) / (||A|| × ||B||)

    # Cria o prompt de QA usando a função definida em prompts.py
    qa_prompt = create_qa_prompt() 

    if "chat_ready" not in st.session_state:
        st.session_state.chat_ready = True

    pergunta = st.text_area("Escreva sua pergunta jurídica", height = 120, placeholder = "Ex.: Quais cláusulas tratam de rescisão e multas?")

    col1, col2 = st.columns([1,1])
    with col1:
        btn = st.button("Perguntar")

    # Executa o pipeline quando o botão for clicado
    if btn:

        if not retriever:
            st.error("Envie um PDF primeiro para habilitar o RAG.")
            st.stop()

        rag_pipeline = create_rag_pipeline(
            retriever=retriever,
            prompt=qa_prompt,
            llm=llm,
        )

        with st.spinner("Gerando resposta…"):
            answer = rag_pipeline.invoke(pergunta)

        st.markdown("### Resposta")
        st.write(answer)   


if __name__ == "__main__":
    main()