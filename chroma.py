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


from transformers import AutoTokenizer, AutoModel
import torch

model_name = "BAAI/bge-large-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def compute_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    return embeddings

import chromadb
from chromadb.config import Settings

def create_vector_database(embeddings, chunks):
    chroma_client = chromadb.Client(Settings())
    collection = chroma_client.create_collection(name="document_chunks")

    ids = [str(idx) for idx in range(len(embeddings))]
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=[{"content": chunk} for chunk in chunks]
    )
    return collection


def query_system(query_text, model, tokenizer, collection, top_k=5):
    inputs = tokenizer(query_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        query_embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()

    results = collection.search(
        query_embedding=query_embedding.tolist(),
        top_k=top_k
    )

    return [doc["content"] for doc in results["documents"]]


# Step 1: Extract text from DOC file
file_path = 'sotr.doc'
text = extract_text_from_doc(file_path)

# Step 2: Split text into chunks
chunks = split_text_into_chunks(text)

# Step 3: Compute embeddings for chunks
embeddings = compute_embeddings(chunks)

# Step 4: Create vector database
collection = create_vector_database(embeddings, chunks)

# Step 5: Query the system
query_text = "What is Min life for bearings"
results = query_system(query_text, model, tokenizer, collection)

for result in results:
    print(result)



