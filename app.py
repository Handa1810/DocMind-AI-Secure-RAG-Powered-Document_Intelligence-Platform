import os
import warnings
import logging
import streamlit as st
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
import hashlib

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

load_dotenv()
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.set_page_config(page_title="📄 Ask Chatbot!", layout="wide")

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

ALLOWED_USERS = {
    "admin": hash_password(os.getenv("ADMIN_PASSWORD")),
    "interviewer": hash_password(os.getenv("INTERVIEWER_PASSWORD")),
    "test": hash_password(os.getenv("TEST_PASSWORD")),
}

def show_login_page():
    st.title("RAG Project by Yash Handa")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in ALLOWED_USERS and ALLOWED_USERS[username] == hash_password(password):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.success("Login Successful!")
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

    st.stop()


def logout():
    if st.sidebar.button("🔓 Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


def show_main_app():

    st.title("DocMind-AI-Secure-RAG-Powered-Document-Intelligence-Platform")

    # Sidebar - Chat History
    st.sidebar.header(f"💬 Chat History (User: {st.session_state.username})")
    logout()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.sidebar.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    for i, message in enumerate(st.session_state.messages):
        with st.sidebar.expander(f"{message['role'].capitalize()} Message {i+1}"):
            st.markdown(message["content"])

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    def compute_file_hashes(files):
        hashes = []
        for file in files:
            content = file.getvalue()
            hashes.append(hashlib.md5(content).hexdigest())
        return tuple(hashes)

    @st.cache_resource(show_spinner="Indexing and Embedding Documents...")
    def create_or_load_vectorstore(files, file_hashes):

        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME")

        if not api_key:
            st.error("❌ Missing PINECONE_API_KEY in .env!")
            return None

        pc = Pinecone(api_key=api_key)

        if index_name not in [idx["name"] for idx in pc.list_indexes()]:
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            st.success(f"🆕 Created Pinecone index: {index_name}")

        index = pc.Index(index_name)

        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        docs = []
        for file in files:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.read())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            docs.extend(loader.load())
            os.remove(tmp_path)

        # Chunking
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = splitter.split_documents(docs)

        # Upload to Pinecone
        vectorstore = PineconeVectorStore.from_documents(
            documents=splits,
            embedding=embedding,
            index_name=index_name
        )

        st.success("📌 Documents embedded into Pinecone successfully.")
        return vectorstore

    vectorstore = None
    if uploaded_files:
        file_hashes = compute_file_hashes(uploaded_files)
        vectorstore = create_or_load_vectorstore(uploaded_files, file_hashes)
        st.success(f"Uploaded {len(uploaded_files)} document(s).")

    # Prompt Templates
    system_prompt = ChatPromptTemplate.from_template(
        """You are highly knowledgeable and precise.
        Answer the user's question clearly and accurately based on the given context.
        Question: {user_prompt}
        Start the answer directly."""
    )

    qa_prompt = PromptTemplate(
        template="Use the context below to answer the question clearly.\n\nContext:\n{context}\n\nQuestion:\n{question}",
        input_variables=["context", "question"]
    )

    # Chat Logic
    prompt = st.chat_input("Ask your question based on the uploaded documents")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            api_key = os.getenv("GROQ_API_KEY")
            model = "llama-3.1-8b-instant"

            if not api_key:
                st.error("❌ Missing GROQ_API_KEY in .env!")
            else:
                groq_chat = ChatGroq(
                    groq_api_key=api_key,
                    model_name=model
                )

                if vectorstore:
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                    qa_chain = load_qa_chain(groq_chat, chain_type="stuff", prompt=qa_prompt)
                    chain = RetrievalQA(
                        combine_documents_chain=qa_chain,
                        retriever=retriever,
                        return_source_documents=True
                    )

                    result = chain({"query": prompt})
                    response = result["result"]

                    st.chat_message("assistant").markdown(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

                    if "source_documents" in result:
                        with st.expander("📚 Retrieved Sources"):
                            for i, doc in enumerate(result["source_documents"]):
                                page = doc.metadata.get("page", "?")
                                st.markdown(f"**Source {i+1} – Page {page}**")
                                st.text(doc.page_content[:500])

                else:
                    chain = system_prompt | groq_chat | StrOutputParser()
                    response = chain.invoke({"user_prompt": prompt})
                    st.chat_message("assistant").markdown(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")

if st.session_state.get("authenticated") is True:
    show_main_app()
else:
    show_login_page()
