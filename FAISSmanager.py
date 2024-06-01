from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

class FAISSManager:
    @staticmethod
    def create_faiss_index(docs, embeddings):
        return FAISS.from_documents(docs, embeddings)
