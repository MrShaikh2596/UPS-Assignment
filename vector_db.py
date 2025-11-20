import os
from dotenv import find_dotenv, load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Debugging to confirm .env file loading
env_path = find_dotenv()
if not env_path:
    raise FileNotFoundError(".env file not found. Please ensure it exists in the project directory.")
print(f"Loading .env file from: {env_path}")

# Load environment variables from .env file
load_dotenv()

# Default PDF to embed when no CLI argument is provided
DEFAULT_PDF_PATH = r"UPS Docs\AI Enginner Use Case Document.pdf"

# Initialize Jina AI embeddings (expects JINA_API_KEY in the environment)
embeddings = JinaEmbeddings(model_name="jina-embeddings-v4")

# Directory where the FAISS index will be stored
INDEX_DIR = "faiss_index"

def resolve_pdf_path(pdf_path):
    """Normalize the provided PDF path and ensure the document exists."""
    if os.path.isabs(pdf_path):
        candidate = pdf_path
    else:
        # Resolve relative paths against the process working directory
        candidate = os.path.abspath(os.path.join(os.getcwd(), pdf_path))

    candidate = os.path.abspath(candidate)
    if not candidate.lower().endswith(".pdf"):
        raise ValueError(f"Provided file is not a PDF: {candidate}")
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"PDF file not found: {candidate}")
    return candidate


def load_single_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

# Chunk documents and add metadata
def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Adjust chunk size as needed
        chunk_overlap=200  # Overlap to maintain context
    )
    chunks = []
    for document in documents:
        doc_chunks = text_splitter.split_documents([document])
        chunks.extend(doc_chunks)
    return chunks

# Use a single FAISS index for all documents
def add_document_to_index(documents):
    chunks = chunk_documents(documents)
    if not chunks:
        print("No chunks generated; nothing to add to the index.")
        return None

    if os.path.isdir(INDEX_DIR):
        vector_store = FAISS.load_local(
            INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        vector_store.add_documents(chunks)
    else:
        vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local(INDEX_DIR)
    print(f"Indexed {len(chunks)} chunks into FAISS store at '{INDEX_DIR}'.")

    return vector_store

# Main function to set up retriever
def setup_retriever(pdf_path):
    resolved_path = resolve_pdf_path(pdf_path)
    print(f"Loading document from {resolved_path}...")
    documents = load_single_pdf(resolved_path)
    print(f"Loaded {len(documents)} pages from the PDF.")

    print("Indexing document...")
    vector_store = add_document_to_index(documents)
    if vector_store is None:
        print("No data indexed. Check the source document.")
    else:
        print("FAISS vector store updated with the provided document.")

    return vector_store

if __name__ == "__main__":
    user_input = input(
        "Enter the PDF path to index (press Enter to use the default UPS document): "
    ).strip()
    target_path = user_input or DEFAULT_PDF_PATH

    retriever = setup_retriever(target_path)
    if retriever is not None:
        print("Vector store build complete.")