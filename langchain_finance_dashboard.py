import os
import streamlit as st
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain_community.vectorstores import FAISS

from Document_Processor import DocumentProcessor
from Embeddings import EmbeddingsManager
from secret_key import openai_key

# Ensure environment variable is set
os.environ['OPENAI_API_KEY'] = openai_key
llm = OpenAI(temperature=0.9, max_tokens=500)
DocumentProcessor = DocumentProcessor()
EmbeddingsManager = EmbeddingsManager()

st.title("News Research tool 📈")

st.sidebar.title("News Articles URL's")

main_placeholder = st.empty()

docs_file_path = "faiss_docs.json"
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)
process_url_clicked = st.sidebar.button("Process URLs")

if process_url_clicked:
    documents_from_urls = DocumentProcessor.load_documents_from_urls(urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = DocumentProcessor.split_documents(documents_from_urls)

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

        main_placeholder.text("Loaded stored FAISS vectorstore...✅✅✅")
    else: 
        main_placeholder.text("No stored vectorstore found. Please process URLs first.")


if vectorstore_openai:
    retriever = vectorstore_openai.as_retriever()


    query = main_placeholder.text_input("Question: ")
    st.header("Answer")
    st.write(query)
    if query: 
        chain = RetrievalQAWithSourcesChain.from_llm(llm = llm, retriever = vectorstore_openai.as_retriever())
        result = chain({"question": query}, return_only_outputs = True)

        st.header("Answer")
        st.write(result["answer"])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)