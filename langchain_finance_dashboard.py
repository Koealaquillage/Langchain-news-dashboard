import os
import streamlit as st
import time
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain_community.vectorstores import FAISS

from Document_Processor import DocumentProcessor
from Embeddings import EmbeddingsManager
from secret_key import openai_key
from PageElements import SideBar, mainElement

# Ensure environment variable is set
os.environ['OPENAI_API_KEY'] = openai_key
llm = OpenAI(temperature=0.9, max_tokens=500)
docs_file_path = "faiss_docs.json"

DocumentProcessor = DocumentProcessor()
EmbeddingsManager = EmbeddingsManager()
SideBar = SideBar()
mainElement = mainElement()

SideBar.create()
mainElement.create()

if SideBar.process_url_clicked:
    documents_from_urls = DocumentProcessor.load_documents_from_urls(SideBar.urls)
    mainElement.placeholder.text("Data Loading...Started...✅✅✅")
    docs = DocumentProcessor.split_documents(documents_from_urls)
    mainElement.placeholder.text("Text Splitter...Started...✅✅✅")
    time.sleep(3)

    vectorstore_openai = FAISS.from_documents(docs, EmbeddingsManager.embeddings)

    DocumentProcessor.save_documents(docs, docs_file_path)

    time.sleep(2)

else:
    # Load the FAISS vectorstore from a file if it exists
    if os.path.exists(docs_file_path):
        documents = DocumentProcessor.load_documents_from_file(docs_file_path)

        embeddings = EmbeddingsManager.embeddings
        vectorstore_openai = FAISS.from_documents(documents, EmbeddingsManager.embeddings)

        mainElement.placeholder.text("Loaded stored FAISS vectorstore...✅✅✅")
    else: 
        mainElement.placeholder.text("No stored vectorstore found. Please process URLs first.")


if vectorstore_openai:
    retriever = vectorstore_openai.as_retriever()


    mainElement.askQuestion()
    if mainElement.query: 
        chain = RetrievalQAWithSourcesChain.from_llm(llm = llm, 
                                                     retriever = vectorstore_openai.as_retriever())
        result = chain({"question": mainElement.query}, return_only_outputs = True)

        mainElement.printAnswer(result)

        mainElement.printSources(result)