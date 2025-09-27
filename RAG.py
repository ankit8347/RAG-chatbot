import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

## Load docs + vector DB
if "vectors" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(
        st.session_state.docs[:50]
    )
    st.session_state.vectors = FAISS.from_documents(
        st.session_state.final_documents, st.session_state.embeddings
    )

st.title("ChatGroq Demo")

# Set API key
os.environ['GROQ_API_KEY'] = 'GROQ_API_KEY'
groq_api_key = os.environ['GROQ_API_KEY']

# ✅ Use normal Groq LLM instead of llama-guard
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

# ✅ Updated prompt (fallback to model knowledge if context missing)
prompt = ChatPromptTemplate.from_template(
"""
Use the following context to answer the question if relevant.  
If the context does not contain the answer, then answer from your own knowledge.  

<context>
{context}
<context>

Question: {input}
"""
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Input box
user_prompt = st.text_input("Input your prompt here")

if user_prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})
    st.write("### Answer:")
    st.write(response['answer'])
    st.write("⏱ Response time:", time.process_time() - start, "seconds")

    # Show retrieved docs
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
