import os
import textract
import faiss
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Initialize the BAAI/bge-large-en model
model_name = "BAAI/bge-large-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Define a function to embed text using the model
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings

# Function to read .doc files and extract text
def read_doc(file_path):
    text = textract.process(file_path).decode('utf-8')
    return text

# Load .doc files from the current directory
document_dir = './'
documents = []
doc_filenames = [f for f in os.listdir(document_dir) if f.endswith('.doc')]
for filename in doc_filenames:
    doc_path = os.path.join(document_dir, filename)
    documents.append(read_doc(doc_path))

# Embed the documents and create a FAISS index
document_embeddings = np.vstack([embed_text(doc) for doc in documents])
index = faiss.IndexFlatL2(document_embeddings.shape[1])
index.add(document_embeddings)

# Embed a query and search for the top 5 similar vectors
query = "What is Min life for bearings"
query_embedding = embed_text(query)
D, I = index.search(query_embedding, 5)

# Retrieve the top 5 similar vectors
top_5_vectors = document_embeddings[I[0]]

# Print the top 5 similar vectors with their distances
for i, (vec, dist) in enumerate(zip(top_5_vectors, D[0])):
    print(f"Top {i+1} vector (distance: {dist}): {vec}")
