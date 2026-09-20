import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.title("RAG Chatbot")

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.session_state.retriever is None:
    uploaded_file = st.file_uploader("Upload a document", type=["pdf"])

    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            data = PyPDFLoader(tmp_path)
            docs = data.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)

            embedding_model = HuggingFaceEmbeddings()

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model
            )

            retriever = vectorstore.as_retriever(
                search_type='mmr',
                search_kwargs={
                    'k': 4,
                    'fetch_k': 10,
                    'lambda_mult': 0.5
                }
            )

            st.session_state.retriever = retriever

        st.rerun()

else:
    llm = ChatGroq(model='openai/gpt-oss-120b')

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
            ),
            (
                "human",
                """Context:
{context}

Question:
{question}
"""
            )
        ]
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    query = st.chat_input("Ask a question about the document")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        docs = st.session_state.retriever.invoke(query)

        context = '\n'.join(
            [doc.page_content for doc in docs]
        )

        final_prompt = prompt_template.invoke({
            'context': context,
            'question': query
        })

        response = llm.invoke(final_prompt)

        st.session_state.messages.append({"role": "assistant", "content": response.content})
        with st.chat_message("assistant"):
            st.write(response.content)