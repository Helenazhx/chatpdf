import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

class InMemoryVecDB:
    def __init__(self, documents = list(), persist_directory = "./vec_db/demo"):
        self.documents = documents
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # 初始化 OpenAIEmbeddings
        if len(documents)>0:
            self.vectordb = self.create_vectordb(documents)

    def create_vectordb(self, documents):
        self.vectordb = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        return self.vectordb

    def search(self, query_embedding, top_k=3):
        # 此方法使用 vectordb 的 similarity_search 来查找最相似的文档
        if self.vectordb is None:
            return "VectorDB not initialized"
        return self.vectordb.similarity_search(query_embedding, top_k)[0].page_content
