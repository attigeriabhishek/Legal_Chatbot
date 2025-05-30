
import pickle
import torch
import streamlit as st
import os
from sentence_transformers import CrossEncoder
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from transformers import AutoTokenizer, AutoModelForCausalLM
# from rerankers import Reranker  # I'll define this for you below

st.set_page_config(page_title="AI Assistant", page_icon="💬", layout="wide")
SAVE_DIR = "vectorstore_data"
MODEL_NAME = "BAAI/bge-large-en-v1.5"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
RERANKER_MODEL = "BAAI/bge-reranker-base"  # Lightweight & accurate
FOLDERS = ["legal_old", "legal_new", "history"]

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name=MODEL_NAME)

embedding_model = load_embedding_model()

@st.cache_resource
def load_vectorstore_and_docs(name):
    path = os.path.join(SAVE_DIR, f"{name}_faiss")
    vectorstore = FAISS.load_local(path, embeddings=embedding_model, allow_dangerous_deserialization=True)

    docs_path = os.path.join(SAVE_DIR, f"{name}_docs.pkl")
    if os.path.exists(docs_path):
        with open(docs_path, "rb") as f:
            docs = pickle.load(f)
    else:
        docs = []
    return vectorstore, docs

vectorstores = {}
documents = {}
for name in FOLDERS:
    vectorstore, docs = load_vectorstore_and_docs(name)
    vectorstores[name] = vectorstore
    documents[name] = docs

@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return tokenizer, model

tokenizer, model = load_llm()

# Load CrossEncoder for reranking
@st.cache_resource
def load_cross_encoder():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

cross_encoder = load_cross_encoder()

def hybrid_retrieval_rerank(name, query, top_k=10):
    dense_retriever = vectorstores[name].as_retriever(search_kwargs={"k": 50})
    bm25_retriever = BM25Retriever.from_documents(documents[name])
    bm25_retriever.k = 50

    retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    
    results = retriever.get_relevant_documents(query)
    
    if name == "history":
        top_k=20
    # Now rerank the results using CrossEncoder
    reranked = rerank(query, results, top_k=top_k)
    return reranked

def rerank(query, documents, top_k=5):
    pairs = [[query, doc.page_content] for doc in documents]
    scores = cross_encoder.predict(pairs)
    doc_score_pairs = list(zip(documents, scores))
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    return [doc for doc, score in doc_score_pairs[:top_k]]

# Generate answer using only the provided context
def generate_answer(query, context, max_words):
    prompt = f"""You are a helpful assistant. You must only use the following context to answer the user's question.
Reframe and rephrase the context to create a clear, coherent, and concise answer.
Do not add any external information.

Context:
{context}

User Question:
{query}

Instructions:
- ONLY use the context to answer.
- Summarize or restructure the context as needed to make the answer better.
- Keep the answer under {max_words} words.
- If the context does not contain enough information, reply: "The answer is not found in the provided documents."

Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,   # Adjust this according to your needs
        temperature=0.4,      # Lower temperature = more focused, less random
        do_sample=True       # Greedy decoding (better for accuracy)
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    final_answer = response.split("Answer:")[-1].strip()

    # Truncate by words if needed
    words = final_answer.split()
    if len(words) > max_words:
        final_answer = " ".join(words[:max_words]) + "..."

    return final_answer

import pandas as pd
import io
def render_answer(answer: str):
    # Detect if answer contains a markdown/pipe table or CSV-like structure
    if "|" in answer and "-" in answer:
        try:
            # Try to convert markdown table to DataFrame
            lines = [line for line in answer.strip().split("\n") if "|" in line]
            table_text = "\n".join(lines)
            df = pd.read_csv(io.StringIO(table_text), sep="|", engine='python', skipinitialspace=True)

            # Drop extra unnamed columns from separators
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            return "table", df
        except Exception:
            pass

    return "text", answer

# st.title("📚 Legal & History Document Search")

# # query = st.text_input("🔍 Enter your query")
# query = st.chat_input("🔍 Enter your query")

# if query:
#     st.markdown("## 🤖 Responses")
#     for name in FOLDERS:
#         with st.spinner(f"🔎 Searching in `{name}`..."):
#             reranked_results = hybrid_retrieval_rerank(name, query, top_k=5)

#             if name.lower() == "history":
#                 # Write results to a text file
#                 with open("reranked_results.txt", "w", encoding="utf-8") as f:
#                     for result in reranked_results:
#                         f.write(str(result) + "\n")

#             context = "\n\n".join([doc.page_content for doc in reranked_results])

#             if context.strip():
#                 answer = generate_answer(query, context, max_words=2000)
#             else:
#                 answer = "❌ No relevant content found."

#             st.subheader(f"📁 {name.capitalize()}")
#             # st.markdown(f"> {answer}")
#             content_type, content = render_answer(answer)
#             if content_type == "table":
#                 st.dataframe(content, use_container_width=True)
#             else:
#                 st.markdown(f"> {content}")
# --- Initialize state ---
if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = True  # Toggle between chatbot and doc search
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("📚 About")
    st.markdown("""
    This AI assistant helps with:
    - Legal document understanding
    - Indian law interpretation
    - Legal procedures and requirements
    - Historical legal context
    """)

    st.subheader("System Info")
    device_info = "GPU" if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        device_info += f" ({torch.cuda.get_device_name(0)})"
    st.text(f"Device: {device_info}")

    st.subheader("Models Used")
    st.markdown(f"""
    - **Embedding**: `{MODEL_NAME}`  
    - **LLM**: `{LLM_MODEL}`  
    - **Reranker**: `{RERANKER_MODEL}`
    """)


    # st.subheader("Tips for Better Search")
    # st.markdown("""
    # - Use specific legal terms
    # - Try full sections (e.g., "Section 302 IPC")
    # - Include year or parties in case queries
    # """)


st.title("📚 Legal & History Document Search")
query = st.chat_input("🔍 Enter your query")

if query:
    st.markdown(f"### 🤖 Responses for: _\"{query}\"_")
    for name in FOLDERS:
        with st.spinner(f"🔎 Searching in `{name}`..."):
            reranked_results = hybrid_retrieval_rerank(name, query, top_k=5)

            if name.lower() == "history":
                with open("reranked_results.txt", "w", encoding="utf-8") as f:
                    for result in reranked_results:
                        f.write(str(result) + "\n")

            context = "\n\n".join([doc.page_content for doc in reranked_results]) if reranked_results else ""

            if context.strip():
                answer = generate_answer(query, context, max_words=2000)
            else:
                answer = "❌ No relevant content found."

            st.subheader(f"📁 {name.capitalize()}")
            content_type, content = render_answer(answer)
            if content_type == "table":
                st.dataframe(content, use_container_width=True)
            else:
                st.markdown(f"> {content}")

