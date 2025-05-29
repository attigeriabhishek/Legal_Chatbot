import logging
import os
from pathlib import Path
from tqdm import tqdm
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import pytesseract
from pdf2image import convert_from_path
import fitz
import time

# Setup logging
logging.basicConfig(level=logging.INFO)

# Folders to process
folders = {
    "legal_new": "data/pdfs/legal_new",  # Your folder paths
    "legal_old": "data/pdfs/legal_old",  # Your folder paths
    "history": "data/pdfs/history"       # Your folder paths
}

# Setup text splitting and embedding model
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")


# Text extraction helper functions
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text = " ".join([page.get_text() for page in doc])
        return text
    except Exception as e:
        logging.error(f"Text extract failed for {pdf_path}: {e}")
        return ""


def extract_text_with_ocr(pdf_path):
    """Extract text from scanned PDFs using OCR (pytesseract)."""
    try:
        logging.info(f"Running OCR on {pdf_path}")
        images = convert_from_path(pdf_path)
        ocr_text = ""
        for img in images:
            ocr_text += pytesseract.image_to_string(img)
        return ocr_text
    except Exception as e:
        logging.error(f"OCR failed for {pdf_path}: {e}")
        return ""


# Process PDFs in a folder
def get_chunks_from_folder(directory):
    """Process all PDFs in a folder, extract text, and create chunks."""
    all_chunks = []
    for pdf_path in tqdm(Path(directory).rglob("*.pdf"), desc=f"Processing {directory}"):
        pdf_path = str(pdf_path)
        text = extract_text_from_pdf(pdf_path)
        
        # If no text extracted, use OCR
        if not text.strip():
            text = extract_text_with_ocr(pdf_path)
        
        # Skip if still no text after OCR
        if not text.strip():
            logging.warning(f"No text found in {pdf_path}, skipping.")
            continue
        
        # Split the extracted text into chunks
        chunks = text_splitter.split_text(text)
        all_chunks.extend(chunks)

    logging.info(f"Total chunks from {directory}: {len(all_chunks)}")
    return all_chunks

from bert_score import score as bert_score
from rouge_score import rouge_scorer

def evaluate_chunks(chunks, reference_texts=None):
    if reference_texts is None:
        reference_texts = chunks  # ← compare chunks to themselves if no references

    evaluation_results = {}

    try:
        P, R, F1 = bert_score(chunks, reference_texts, lang="en", verbose=True)
        evaluation_results['BERTScore'] = {
            'Precision': P.mean().item(),
            'Recall': R.mean().item(),
            'F1': F1.mean().item()
        }
    except Exception as e:
        evaluation_results['BERTScore'] = {'Error': str(e)}
        logging.warning(f"BERTScore evaluation failed: {e}")

    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for chunk, ref in zip(chunks, reference_texts):
            score = scorer.score(ref, chunk)
            for key in rouge_scores:
                rouge_scores[key].append(score[key].fmeasure)
        
        evaluation_results['ROUGE'] = {
            'ROUGE-1': sum(rouge_scores['rouge1']) / len(rouge_scores['rouge1']),
            'ROUGE-2': sum(rouge_scores['rouge2']) / len(rouge_scores['rouge2']),
            'ROUGE-L': sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL'])
        }
    except Exception as e:
        evaluation_results['ROUGE'] = {'Error': str(e)}
        logging.warning(f"ROUGE evaluation failed: {e}")

    return evaluation_results


# Process the folder and generate embeddings
def process_with_langchain(folder, name):
    docs = []
    
    # Load PDFs from the folder
    for path in tqdm(Path(folder).rglob("*.pdf"), desc=f"LangChain Processing {folder}"):
        loader = PyMuPDFLoader(str(path))
        try:
            docs.extend(loader.load())
            logging.info(f"Loaded {len(docs)} documents from {path}")
        except Exception as e:
            logging.error(f"Failed to load {path}: {e}")
    
    logging.info(f"{len(docs)} documents loaded from {folder}.")
    
    # Check if any documents were loaded
    if not docs:
        logging.error("No documents were loaded. Check if the PDFs are in the correct format and path.")
        return [], None

    # Split documents into chunks
    # split_docs = text_splitter.split_documents(docs)
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
    )

    split_docs = splitter.split_documents(docs)

    logging.info(f"Total chunks created from {folder}: {len(split_docs)}.")
    
    # Check if chunks are empty
    if not split_docs:
        logging.error("No chunks were created. Check the text splitter configuration and the document content.")
        return [], None

    # Create embeddings
    try:
        logging.info("🔵 Starting FAISS embedding creation...")
        start_time = time.time()
        vectorstore = FAISS.from_documents(split_docs, embedding_model)
        end_time = time.time()
        total_time = end_time - start_time

        with open(os.path.join(SAVE_DIR, f"{name}_docs.pkl"), "wb") as f:
            pickle.dump(split_docs, f)

        logging.info(f"✅ Embeddings created and stored with FAISS for {folder}.")
        logging.info(f"⏱️ Time taken to create embeddings and store in FAISS: {total_time:.2f} seconds.")

        return split_docs, vectorstore
    except Exception as e:
        logging.error(f"Error during FAISS indexing: {e}")
        return [], None

# Function to load FAISS vectorstore later
def load_vectorstore(name):
    """Load a saved FAISS vectorstore from disk."""
    db_path = f"vectorstore_data/{name}"
    if os.path.exists(db_path):
        return FAISS.load_local(db_path, embedding_model)
    else:
        logging.warning(f"Vectorstore for {name} not found.")
        return None

# Main execution
import os
import pickle

SAVE_DIR = "vectorstore_data"
os.makedirs(SAVE_DIR, exist_ok=True)

langchain_data = {}

for name, folder in folders.items():
    logging.info(f"\n--- LangChain Processing: {name} ---")
    chunks, vectorstore = process_with_langchain(folder, name)

    if not chunks or not vectorstore:
        continue

    # Evaluate
    metrics = evaluate_chunks([doc.page_content for doc in chunks])
    
    # Save FAISS vectorstore
    vectorstore.save_local(f"{SAVE_DIR}/{name}_faiss")

    # Save chunks using pickle
    with open(f"{SAVE_DIR}/{name}_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    # Save metrics
    with open(f"{SAVE_DIR}/{name}_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)

    # Store in memory (optional)
    langchain_data[name] = {
        "chunks": chunks,
        "vectorstore": vectorstore,
        "metrics": metrics
    }

    logging.info(f"✅ Finished processing {name}. Metrics: {metrics}")
    # load_vectorstore(name)

