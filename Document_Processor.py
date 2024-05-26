import json
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

class DocumentProcessor:
    def __init__(self, chunk_size=1000):
        self.chunk_size = chunk_size

    def load_documents_from_urls(self, urls):
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()
        return data

    def split_documents(self, data):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ","],
            chunk_size=self.chunk_size
        )
        docs = text_splitter.split_documents(data)
        return docs

    def save_documents(self, docs, file_path):
        docs_data = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        with open(file_path, "w") as f:
            json.dump(docs_data, f)

    def load_documents_from_file(self, file_path):
        with open(file_path, "r") as f:
            docs_data = json.load(f)

        documents = [
        Document(page_content=doc["content"], metadata=doc["metadata"])
        for doc in docs_data
    ]
        return documents
