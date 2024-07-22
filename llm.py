import textract
from langchain_community.document_loaders import PyPDFDirectoryLoader
import pandas as pd
from openpyxl import load_workbook
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import torch
import ollama
import logging
from datetime import datetime

# Set up logging
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = f"debug_log_{current_time}.log"
logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s %(message)s')

DATA_PATH = "./data/"

def extract_text_from_doc(file_path):
    logging.debug(f"Extracting text from doc: {file_path}")
    text = textract.process(file_path).decode('utf-8')
    return text

def load_documents():
    logging.debug(f"Loading documents from directory: {DATA_PATH}")
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_text_into_chunks(text, chunk_size=500):
    logging.debug("Splitting text into chunks")
    words = re.findall(r'\w+|\s+|[^\w\s]', text)
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_length += len(word)
        if current_length > chunk_size:
            chunks.append(''.join(current_chunk))
            current_chunk = []
            current_length = len(word)
        current_chunk.append(word)

    if current_chunk:
        chunks.append(''.join(current_chunk))

    logging.debug(f"Total chunks created: {len(chunks)}")
    return chunks

model = SentenceTransformer("BAAI/bge-large-en")
logging.debug("SentenceTransformer model loaded")

def compute_embeddings(chunks):
    logging.debug(f"Computing embeddings for {len(chunks)} chunks")
    embeddings = model.encode(chunks)
    return embeddings

def create_vector_database(embeddings):
    logging.debug("Creating vector database")
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

def query_system(query_text, model, index, chunks, top_k=10):
    logging.debug(f"Querying system with text: {query_text}")
    query_embedding = model.encode([query_text])
    distances, indices = index.search(query_embedding, top_k)
    results = [chunks[i] for i in indices[0]]
    return results

def classify_text_with_llm(query, context):
    logging.debug(f"Classifying text with LLM, query: {query}")
    prompt = f"Given the context: '{context}'\nIs the following query: '{query}' ACC or DEV? Respond with either 'ACC' or 'DEV'."
    response = ollama.generate(model = "mistral:latest", prompt = prompt)
    return response['response']

# Main script
logging.debug("Main script started")
file_path_excel = 'extracted_points_1.xlsx'
df = pd.read_excel(file_path_excel, sheet_name='Extracted Points')

if df.shape[1] < 3:
    df.insert(2, 'Result', '')
if df.shape[1] < 4:
    df.insert(3, 'Classification', '')

file_path_doc = 'bid.docx'
text = extract_text_from_doc(file_path_doc)
chunks = split_text_into_chunks(text)
embeddings = compute_embeddings(chunks)
index = create_vector_database(embeddings)

a = 0

for i, query in enumerate(df.iloc[:, 1]):
    if a < 83:
        a += 1
        continue   

    results = query_system(query, model, index, chunks)
    top_results = " | ".join(results)
    df.iloc[i, 2] = top_results
    
    # Classification step
    classification = classify_text_with_llm(df.iloc[i, 0], top_results)
    df.iloc[i, 3] = classification
    logging.debug(f"Classification for point {i+1}: {classification}")
    a += 1
    if a == 86:
        break

df.to_excel(file_path_excel, sheet_name='Extracted Points', index=False)
logging.debug("Processing complete. Results have been stored in the Excel file.")
