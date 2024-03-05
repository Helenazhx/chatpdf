from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter

def extract_text_from_pdf(filename):
    loader = PyPDFLoader(filename)
    pages = loader.load()
    text_splitter = TokenTextSplitter(chunk_size=35, chunk_overlap=5)
    # text_splitter = RecursiveCharacterTextSplitter(
    # chunk_size = 1500,
    # chunk_overlap = 150)
    docs = text_splitter.split_documents(pages)
    return docs