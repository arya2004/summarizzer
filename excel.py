import textract
from langchain_community.document_loaders import PyPDFDirectoryLoader
import pandas as pd
from openpyxl import load_workbook

DATA_PATH = "./data/"

def extract_text_from_doc(file_path):
    text = textract.process(file_path).decode('utf-8')
    return text

def load_documents():
    """
    Load PDF documents from the specified directory using PyPDFDirectoryLoader.
    Returns:
    List of Document objects: Loaded PDF documents represented as Langchain Document objects.
    """
    # Initialize PDF loader with specified directory
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    # Load PDF documents and return them as a list of Document objects
    return document_loader.load()

import re

def split_text_into_chunks(text, chunk_size=500):
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

    return chunks

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings

import faiss

def create_vector_database(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

def query_system(query_text, model, index, chunks, top_k=3):
    query_embedding = model.encode([query_text])
    distances, indices = index.search(query_embedding, top_k)
    results = [chunks[i] for i in indices[0]]
    return results

# Load the Excel file
file_path_excel = 'extracted_points_10.xlsx'
df = pd.read_excel(file_path_excel, sheet_name='Extracted Points')

# Ensure the third column exists
if df.shape[1] < 3:
    df.insert(2, 'Result', '')

# Step 1: Extract text from DOC file
file_path_doc = 'bid.docx'
text = extract_text_from_doc(file_path_doc)

# Step 2: Split text into chunks
chunks = split_text_into_chunks(text)

# Step 3: Compute embeddings for chunks
embeddings = compute_embeddings(chunks)

# Step 4: Create vector database
index = create_vector_database(embeddings)

# Step 5: Process each query in the Excel file and store the result
for i, query in enumerate(df.iloc[:, 1]):  # Assuming queries are in the second column
    results = query_system(query, model, index, chunks)
    top_results = " | ".join(results)  # Join top 3 results with a separator
    df.iloc[i, 2] = top_results  # Storing the results in the third column

# Save the updated DataFrame back to the Excel file
df.to_excel(file_path_excel, sheet_name='Extracted Points', index=False)

print("Processing complete. Results have been stored in the Excel file.")
