
# Simple RAG Chatbot

Try my chatbot here: [Simple RAG Chatbot](https://lvdehrr4xdtqvsjapp5idsx.streamlit.app/)

This chatbot is built using a simple **Retrieval-Augmented Generation (RAG)** pipeline. The process includes document loading, splitting the data into chunks, generating vector embeddings, and storing these embeddings in a vector database. In this application, the vector database is stored locally on the device's disk.

When a user generates a query, the retriever identifies the embedding vectors most similar to the query vector. The retrieved context is then passed through a prompt template to the LLM, which generates an answer based on the relevant information.
