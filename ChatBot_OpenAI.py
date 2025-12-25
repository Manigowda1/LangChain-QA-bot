import streamlit as st
from PyPDF2 import PdfReader
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
OPENAI_API_KEY = ""
st.header("Personal_file_Intelligent Chat Bot")

# Upload files
with st.sidebar:
    st.title("Welcome to Interactive Chat Bot")
    file = st.file_uploader("Upload a file and start asking questions",type="pdf")

#Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        # st.write(text)

#Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    # st.write(chunks)

    #Generate embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small",api_key=OPENAI_API_KEY)

    #Vectore store - FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

    #Get User Query
    user_query = st.text_input("Type your Query")
    #Do Similarity search
    if user_query:
        match = vector_store.similarity_search(user_query)
        # st.write(match)

        #define the LLM
        llm = ChatOpenAI(
            open_api_key = OPENAI_API_KEY,
            temperature = 0 ,
            max_tokens= 1000,
            model_name = "gpt-3.5-turbo"
        )

        #Output results
        chain = load_qa_chain(llm,chain_type="stuff")
        response = chain.run(input_documents = match , question = user_query)
        st.write(response)