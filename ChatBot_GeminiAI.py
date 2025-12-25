import streamlit as st
from PyPDF2 import PdfReader
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import os

GOOGLE_API_KEY = ""  # your Gemini API key

st.header("Personal_file_Intelligent Chat Bot")

# Upload files
with st.sidebar:
    st.title("Welcome to Interactive Chat Bot")
    file = st.file_uploader("Upload a file and start asking questions", type="pdf")

# Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Generate embeddings with Gemini
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    # Vector store - FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Get User Query
    user_query = st.text_input("Type your Query")

    if user_query:
        # Do Similarity search
        match = vector_store.similarity_search(user_query)

        # Define the Gemini LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",   # or "gemini-1.5-flash" for faster responses
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
            max_output_tokens=1000
        )

        # Output results
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=match, question=user_query)
        st.write(response)
