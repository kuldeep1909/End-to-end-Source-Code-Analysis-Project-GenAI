from src.helper import repo_ingestion, load_embeddings, load_repo, text_splitter
from dotenv import load_dotenv
import os
from langchain.vectorstores import Chroma

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
os.environ['GROQ_API_KEY'] = GROQ_API_KEY

## url = url of any github repo we wann use this model for

# repo_ingestion(url)

documents = load_repo("repo/")
text_chunks = text_splitter(documents)
embeddings  = load_embeddings()


## store the vector in the Chroma database

vectorstore = Chroma.from_documents(text_chunks, embedding=embeddings, persist_directory=".db/")
vectorstore.persist()