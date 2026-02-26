import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Resume RAG Chatbot")
st.title("📄 Resume Screening RAG Chatbot")

uploaded_files = st.file_uploader(
    "Upload Resume PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    documents = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            loader = PyPDFLoader(tmp_file.name)
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = uploaded_file.name

            documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    # FREE embeddings (local)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Groq LLM
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant"
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    query = st.text_input("Ask about candidates")

    if query:
        response = qa_chain({"query": query})

        st.subheader("Answer:")
        st.write(response["result"])

        st.subheader("Matched Resume Sources:")
        for doc in response["source_documents"]:
            st.write(doc.metadata["source"])