from flask import Flask, request, jsonify
from groq import Groq
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from duckduckgo_search import DDGS
import PyPDF2
import io
import os

load_dotenv()

app = Flask(__name__)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
vectorstore = None
conversation_history = []

def web_search(query):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                return "\n".join([f"{r['title']}: {r['body']}" for r in results])
        return "No results found."
    except Exception as e:
        return f"Search failed: {str(e)}"

def search_documents(question):
    if not vectorstore:
        return None
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(question)
        if docs:
            return "\n\n".join([doc.page_content for doc in docs])
        return None
    except Exception as e:
        return None

def needs_web_search(question):
    web_keywords = ["latest", "current", "today", "news", "2025", "2026", "trending", 
                   "recent", "now", "job market", "salary", "price", "who is", "when did"]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in web_keywords)

def run_agent(user_question):
    global conversation_history

    doc_context = search_documents(user_question)
    web_context = None

    if needs_web_search(user_question) or not doc_context:
        web_context = web_search(user_question)

    system_prompt = "You are a smart AI assistant for Charan, an aspiring AI Engineer. Be direct and specific."

    context_parts = []
    if doc_context:
        context_parts.append(f"From uploaded document:\n{doc_context}")
    if web_context:
        context_parts.append(f"From web search:\n{web_context}")

    full_context = "\n\n".join(context_parts) if context_parts else ""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history[-6:])

    user_content = f"{full_context}\n\nQuestion: {user_question}" if full_context else user_question
    messages.append({"role": "user", "content": user_content})

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3
        )
        answer = response.choices[0].message.content
        conversation_history.append({"role": "user", "content": user_question})
        conversation_history.append({"role": "assistant", "content": answer})
        if len(conversation_history) > 12:
            conversation_history = conversation_history[-12:]
        source = "document" if doc_context and not web_context else "web" if web_context and not doc_context else "document + web"
        return answer, source
    except Exception as e:
        return f"Error: {str(e)}", "error"

@app.route("/")
def home():
    return open("index.html", encoding="utf-8").read()

@app.route("/upload", methods=["POST"])
def upload():
    global vectorstore
    text = request.json["text"]
    name = request.json["name"]
    chunks = text_splitter.split_text(text)
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return jsonify({"status": "ok", "chunks": len(chunks), "name": name})

@app.route("/upload-pdf", methods=["POST"])
def upload_pdf():
    global vectorstore
    file = request.files["pdf"]
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    chunks = text_splitter.split_text(text)
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return jsonify({
        "status": "ok",
        "chunks": len(chunks),
        "name": file.filename,
        "pages": len(pdf_reader.pages)
    })

@app.route("/chat", methods=["POST"])
def chat():
    question = request.json.get("message", "")
    if not question:
        return jsonify({"error": "No message provided"}), 400
    try:
        answer, source = run_agent(question)
        return jsonify({"reply": answer, "source": source})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/clear", methods=["POST"])
def clear():
    global conversation_history
    conversation_history = []
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(debug=True)