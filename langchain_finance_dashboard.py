import os
import streamlit as st
import time
import json
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from secret_key import openai_key

# Ensure environment variable is set
os.environ['OPENAI_API_KEY'] = openai_key
llm = OpenAI(temperature=0.9, max_tokens=500)

def save_documents(docs, docs_file_path):
    docs_data = [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in docs
    ]
    with open(docs_file_path, "w") as f:
        json.dump(docs_data, f)

def load_faiss_index_from_documents(docs_file_path):
    with open(docs_file_path, "r") as f:
        docs_data = json.load(f)
    
    embeddings = OpenAIEmbeddings()
    
    documents = [
        Document(page_content=doc["content"], metadata=doc["metadata"])
        for doc in docs_data
    ]
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

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
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n", ".", ","],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()

    time.sleep(3)

    vectorstore_openai = FAISS.from_documents(docs, embeddings)

    save_documents(docs, docs_file_path)

    time.sleep(2)

else:
    # Load the FAISS vectorstore from a file if it exists
    if os.path.exists(docs_file_path):
        vectorstore_openai = load_faiss_index_from_documents(docs_file_path)
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