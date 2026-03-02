import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# -------------------------------
# Load Environment Variables
# -------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="College Notes Assistant", page_icon="📚")
st.title("📚 College Notes Assistant (RAG + Groq)")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in .env file")
    st.stop()


# -------------------------------
# Upload Section
# -------------------------------
uploaded_file = st.file_uploader("Upload your PDF notes", type="pdf")
question = st.text_input("Ask a question from the notes")


# -------------------------------
# PDF Processing Function
# -------------------------------
@st.cache_resource
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = splitter.split_documents(docs)

    # Faster embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(split_docs, embeddings)

    return vectorstore


# -------------------------------
# Main Execution
# -------------------------------
if uploaded_file and question:
    try:
        with st.spinner("Processing... Please wait ⏳"):

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp:
                tmp.write(uploaded_file.read())
                file_path = tmp.name

            # Process PDF
            vectorstore = process_pdf(file_path)

            # Initialize Groq LLM
            llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name="llama-3.1-8b-instant",
                temperature=0
            )

            # Prompt Template
            prompt = ChatPromptTemplate.from_template(
                """
                Answer the question using only the context below.
                If the answer is not found,
                say "Answer not found in the document."

                Context:
                {context}

                Question:
                {input}
                """
            )

            # Create RAG Chain
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = vectorstore.as_retriever()
            retrieval_chain = create_retrieval_chain(
                retriever,
                document_chain
            )

            # Generate Answer
            response = retrieval_chain.invoke(
                {"input": question}
            )

            st.success("✅ Answer:")
            st.write(response["answer"])

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
