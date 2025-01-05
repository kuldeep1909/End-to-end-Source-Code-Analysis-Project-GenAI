# all the utility function 
import os
from git import Repo   # for this we take the link (GitPython) to clone the repository from the github
from langchain.text_splitter import Language   # language detection
from langchain_community.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import  HuggingFaceEmbeddings

from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain


## cloning the repo
def repo_ingestion(repo_url):
    os.makedirs("repo", exist_ok=True)
    repo_path = "repo/"
    Repo.clone_from(repo_url, to_path=repo_path)



## loading the reporistorirs as the documents
def load_repo(repo_path):
    # load all the code from the repo
    loader = GenericLoader.from_filesystem(repo_path,
                                       glob = "**/*",
                                       suffixes=[".py"],
                                       parser=LanguageParser(language= Language.PYTHON, parser_threshold=500))
    documents = loader.load()

    return documents


## create the text chunks
def text_splitter(documents):
    # context aware splittig of the code
    documents_splitter = RecursiveCharacterTextSplitter.from_language(language= Language.PYTHON,
                                                                  chunk_size= 500,
                                                                  chunk_overlap= 20)
    
    text_chunks = documents_splitter.split_documents(documents)

    return text_chunks


## loading the embedding model
def load_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="microsoft/codebert-base")
    return embeddings

