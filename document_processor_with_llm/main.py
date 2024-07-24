import textract
from langchain_community.document_loaders import PyPDFDirectoryLoader
import pandas as pd
from openpyxl import load_workbook
import re
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import numpy as np
import faiss
import torch
import logging
from datetime import datetime

import config

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = config.OPENAI_API_KEY
openai_api_base = config.OPENAI_API_BASE

# Configuration Variables
data_path = config.DATA_PATH
chunk_size = config.CHUNK_SIZE
model_name = config.MODEL_NAME
top_k = config.TOP_K
excel_file_path = config.EXCEL_FILE_PATH
doc_file_path = config.DOC_FILE_PATH
log_file_prefix = config.LOG_FILE_PREFIX
llm_model = config.LLM_MODEL

# Set up logging
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = f"{log_file_prefix}{current_time}.log"
logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s %(message)s')

def extract_text_from_doc(file_path):
    logging.debug(f"Extracting text from doc: {file_path}")
    text = textract.process(file_path).decode('utf-8')
    return text

def load_documents():
    logging.debug(f"Loading documents from directory: {data_path}")
    document_loader = PyPDFDirectoryLoader(data_path)
    return document_loader.load()

def split_text_into_chunks(text, chunk_size=chunk_size):
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

model = SentenceTransformer(model_name)
logging.debug("SentenceTransformer model loaded")

keybert_model = KeyBERT(model="intfloat/multilingual-e5-base")
logging.debug("KeyBERT model loaded")

def compute_embeddings(chunks):
    logging.debug(f"Computing embeddings for {len(chunks)} chunks")
    embeddings = model.encode(chunks)
    return embeddings

def extract_keywords(chunks):
    logging.debug("Extracting keywords from chunks")
    metadata = []
    for chunk in chunks:
        keywords = keybert_model.extract_keywords(chunk, keyphrase_ngram_range=(1, 2), stop_words='english')
        metadata.append(keywords)
    logging.debug(f"Extracted metadata for {len(chunks)} chunks")
    return metadata

def create_vector_database(embeddings, metadata):
    logging.debug("Creating vector database")
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index, metadata

def optimize_query_with_llm(query_text):
    logging.debug(f"Optimizing query text with LLM: {query_text}")

    prompt = (
        f"Optimize the following search query for better search results:\n\n"
        f"Original Query: {query_text}\n\n"
        f"Optimized Query:"
    )

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Your task is to optimize the given search query for better search results."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=model_gpt,
        )

        optimized_query = chat_completion.choices[0].message.content.strip()
        logging.debug(f"Optimized query: {optimized_query}")

        return optimized_query
    except Exception as e:
        logging.error(f"Error during query optimization with LLM: {e}")
        return query_text

def query_system(query_text, model, index, chunks, metadata, top_k=top_k):
    logging.debug(f"Querying system with text: {query_text}")
    query_embedding = model.encode([query_text])
    distances, indices = index.search(query_embedding, top_k)
    results = [chunks[i] for i in indices[0]]
    result_metadata = [metadata[i] for i in indices[0]]
    return results, result_metadata

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model_gpt = models.data[0].id

def classify_text_with_llm(query, context):
    logging.debug(f"Classifying text with LLM")

    prompt = (
        f"Given the following context:\n'{context}'\n"
        f"Please determine if the following query is compliant. "
        f"Respond with either 'ACC' (Accepted/Compliant) or 'DEV' (Deviated/Not Compliant). "
        f"If there are any comments, tips, or reviews provided, include those as well without hallucination. "
        f"Query: '{query}'"
    )

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Your task is to determine if the query is compliant based on the given context. Respond with either 'ACC' or 'DEV'. If there are any comments, tips, or reviews provided, include those as well without hallucination."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=model_gpt,
        )

        response = chat_completion.choices[0].message.content.strip()
        logging.debug(f"LLM response: {response}")

        return response
    except Exception as e:
        logging.error(f"Error during LLM classification: {e}")
        return "Error"

# Main script
logging.debug("Main script started")
df = pd.read_excel(excel_file_path, sheet_name='Extracted Points')

if df.shape[1] < 3:
    df.insert(2, 'Result', '')
if df.shape[1] < 4:
    df.insert(3, 'Classification', '')

text = extract_text_from_doc(doc_file_path)
chunks = split_text_into_chunks(text)
embeddings = compute_embeddings(chunks)
metadata = extract_keywords(chunks)
index, metadata = create_vector_database(embeddings, metadata)

a = 0

for i, query in enumerate(df.iloc[:, 1]):
    # Optimize the query with LLM
    optimized_query = optimize_query_with_llm(query)
    logging.debug(f"Optimized query for point {i+1}: {optimized_query}")

    # Query the system with the optimized query
    results, result_metadata = query_system(optimized_query, model, index, chunks, metadata)
    top_results = " | ".join(results)
    df.iloc[i, 2] = top_results
    
    # Classification step
    classification = classify_text_with_llm(df.iloc[i, 0], top_results)
    df.iloc[i, 3] = classification
    logging.debug(f"Classification for point {i+1}: {classification}")
    a += 1
    if a > 10:
        break

df.to_excel(excel_file_path, sheet_name='Extracted Points', index=False)
logging.debug("Processing complete. Results have been stored in the Excel file.")
