import os
import tempfile

import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain.agents.agent_toolkits import (
    VectorStoreInfo,
    VectorStoreToolkit,
    create_vectorstore_agent,
)
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv(find_dotenv())

# Set the title and subtitle of the app
st.title("ChatPDF")
st.subheader("Chat with your PDF documents using the power of OpenAI's GPT-4 model.")

# Allow the user to upload a PDF document
st.header("Upload a PDF document")
doc = st.file_uploader("Document", type=(["pdf"]), label_visibility="hidden")
if doc is not None:
    temp_dir = tempfile.TemporaryDirectory()
    temp_file_path = os.path.join(temp_dir.name, doc.name)
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(doc.read())
    st.write(doc.name, "uploaded successfully.")

# Set up PDF chat capabilities
if doc is not None:
    st.header("Chat with your PDF document")

    # Create instance of OpenAI LLM and OpenAI Embeddings
    llm = ChatOpenAI(model="gpt-4", temperature=0.1, verbose=True)
    embeddings = OpenAIEmbeddings()

    # Load PDF document into vector database
    loader = PyPDFLoader(temp_file_path)
    pages = loader.load_and_split(
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    )
    store = Chroma.from_documents(pages, embeddings, collection_name="PDF")
    vectorstore_info = VectorStoreInfo(
        name="PDF",
        description="A PDF document to answer users' questions",
        vectorstore=store,
    )
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

    # Create agent that can retrieve data from PDF document in vector database
    agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True)

# Allow the user to chat with their PDF document
if doc is not None:
    # Allow the user to ask a question
    prompt = st.text_input("What would you like to ask?")
    if prompt:
        # Display the response
        response = agent_executor.run(prompt)
        st.write(response)

        # Add an expander to display the most similar pages
        with st.expander("Document Similarity Search"):
            search = store.similarity_search_with_score(prompt)
            st.write(search[0][0].page_content)
