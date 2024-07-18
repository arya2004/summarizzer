# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Langchain dependencies
from langchain_community.document_loaders import PyPDFDirectoryLoader  # Importing PDF loader from Langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Importing text splitter from Langchain
from langchain.schema import Document  # Importing Document schema from Langchain
from langchain_community.vectorstores.chroma import Chroma  # Importing Chroma vector store from Langchain
from dotenv import load_dotenv  # Importing dotenv to get API key from .env file
import os  # Importing os module for operating system functionalities
import shutil  # Importing shutil module for high-level file operations

from sentence_transformers import SentenceTransformer

DATA_PATH = "./data/"

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

documents = load_documents()  # Call the function
# Inspect the contents of the first document as well as metadata
print(documents[0])

def split_text(documents):
    """
    Split the text content of the given list of Document objects into smaller chunks.
    Args:
    documents (list[Document]): List of Document objects containing text content to split.
    Returns:
    list[Document]: List of Document objects representing the split text chunks.
    """
    # Initialize text splitter with specified parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Size of each chunk in characters
        chunk_overlap=100,  # Overlap between consecutive chunks
        length_function=len,  # Function to compute the length of the text
        add_start_index=True,  # Flag to add start index to each chunk
    )

    # Split documents into smaller chunks using text splitter
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Print example of page content and metadata for a chunk
    document = chunks[0]
    print(document.page_content)
    print(document.metadata)

    return chunks  # Return the list of split text chunks

# Path to the directory to save Chroma database
CHROMA_PATH = "chroma"

def save_to_chroma(chunks):
    """
    Save the given list of Document objects to a Chroma database.
    Args:
    chunks (list[Document]): List of Document objects representing text chunks to save.
    Returns:
    None
    """
    # Clear out the existing database directory if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Initialize the all-MiniLM-L6-v2 embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Compute embeddings for each chunk
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts)

    # Create a new Chroma database
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=model.encode)

    # Add documents and embeddings to the Chroma database
    for i, text in enumerate(texts):
        db.add_texts([text], embeddings=[embeddings[i]], metadatas=[chunks[i].metadata])

    # Persist the database to disk
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def generate_data_store():
    """
    Function to generate vector database in chroma from documents.
    """
    documents = load_documents()  # Load documents from a source
    chunks = split_text(documents)  # Split documents into manageable chunks
    save_to_chroma(chunks)  # Save the processed data to a data store

# Load environment variables from a .env file
load_dotenv()
# Generate the data store
generate_data_store()

query_text = "What is Min life for bearings"

def query_rag(query_text):
    """
    Query a Retrieval-Augmented Generation (RAG) system using Chroma database and all-MiniLM-L6-v2.
    Args:
    - query_text (str): The text to query the RAG system with.
    Returns:
    - formatted_response (str): Formatted response including the generated text and sources.
    - response_text (str): The generated response text.
    """
    # Initialize the all-MiniLM-L6-v2 embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Prepare the database
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=model.encode)

    # Compute the query embedding
    query_embedding = model.encode([query_text])

    # Retrieving the context from the DB using similarity search
    results = db.similarity_search_with_relevance_scores(query_embedding, k=3)

    # Check if there are any matching results or if the relevance score is too low
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return []

    # Combine context from matching documents
    return results

# Let's call our function we have defined
results = query_rag(query_text)
# and finally, inspect our final response!
print(results)
