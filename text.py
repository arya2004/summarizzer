import textract

def extract_text_from_doc(file_path):
    text = textract.process(file_path).decode('utf-8')
    return text

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


def query_system(query_text, model, index, chunks, top_k=5):
    query_embedding = model.encode([query_text])
    distances, indices = index.search(query_embedding, top_k)
    results = [chunks[i] for i in indices[0]]
    return results


# Step 1: Extract text from DOC file
file_path = 'sotr.doc'
text = extract_text_from_doc(file_path)

# Step 2: Split text into chunks
chunks = split_text_into_chunks(text)

# Step 3: Compute embeddings for chunks
embeddings = compute_embeddings(chunks)

# Step 4: Create vector database
index = create_vector_database(embeddings)

# Step 5: Query the system
query_text = "What is Min life for bearings"
results = query_system(query_text, model, index, chunks)

for result in results:
    print(result)
