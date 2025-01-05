from flask import Flask, request, render_template, jsonify

from langchain.vectorstores import Chroma
from src.helper import load_embeddings
from dotenv import load_dotenv
import os
from src.helper import repo_ingestion
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq



app = Flask(__name__)

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
os.environ['GROQ_API_KEY'] = GROQ_API_KEY


embeddings = load_embeddings()
persist_directory = "db"
# now we can load the persisted database form disk , and use it as normal
vectorstore = Chroma(persist_directory=persist_directory, 
                     embedding_function=embeddings)



# create the llm chain

llm = ChatGroq(
    groq_api_key = GROQ_API_KEY,
    model_name = "llama-3.3-70b-specdec")

# memory
memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)

## Create the final chian
qa = ConversationalRetrievalChain.from_llm(llm, retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":8}), memory=memory)


# create two route from flask 
# first for renderging theindex.html file

@app.route('/', methods=["GET", "POST"])
def index():
    return render_template("index.html")


# route for the repo ingestion 
@app.route('/chatbot', methods={"GET", "POST"})
def gitrepo():

    if request.method == "POST":
        user_input = request.form["question"]
        repo_ingestion(user_input)
        os.system("python store_index.py")

    return jsonify({"response": str(user_input)})


## route for the chat operation 
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)

    if input == "clear":
        os.system("rm -rf repo")
    
    result = qa(input)
    print(result['answer'])
    return str(result['answer'])



if __name__ == "__main__":
    app.route(host = "0.0.0.0", port = 8080, debug=True)