import numpy as np
from langchain_openai import OpenAIEmbeddings

class EmbeddingsManager:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()

    def generate_embeddings(self, docs):
        return self.embeddings.embed_documents([doc.page_content for doc in docs])

    def save_embeddings(self, embeddings, file_path):
        np.save(file_path, np.array(embeddings))

    def load_embeddings(self, file_path):
        return np.load(file_path, allow_pickle=False)
