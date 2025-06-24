import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
import streamlit as st

load_dotenv()

# Set up Groq LLM (e.g., llama3-8b)
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"],
    model_name="deepseek-r1-distill-llama-70b",  # or "llama3-70b-8192" if you're feeling spicy
    temperature=0.2
)

def load_documents(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            docs.extend(loader.load())
    return docs

def create_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embed_model)
    return vectorstore

def generate_swot(retriever, company_name):
    prompt = open("prompts/swot_template.txt").read().replace("{company_name}", company_name)

    # Get only chunks relevant to this company
    refined_docs = retriever.get_relevant_documents(f"Details about {company_name}")

    # Create a "stuff" chain using your Groq-powered LLM
    qa_chain = load_qa_chain(llm=llm, chain_type="stuff")

    # Run the chain with the refined docs + SWOT prompt
    return qa_chain.run(input_documents=refined_docs, question=prompt)