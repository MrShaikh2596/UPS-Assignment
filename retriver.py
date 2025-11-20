import os

from dotenv import find_dotenv, load_dotenv
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment configuration early so API keys are available
env_path = find_dotenv()
if not env_path:
    raise FileNotFoundError(
        ".env file not found. Please ensure it exists in the project directory."
    )
print(f"Loading .env file from: {env_path}")
load_dotenv()

# Directory where the FAISS index is stored
INDEX_DIR = "faiss_index"

# Initialize shared clients
embeddings = JinaEmbeddings(model_name="jina-embeddings-v4")


def load_vector_store(index_dir: str = INDEX_DIR) -> FAISS:
    """Load the FAISS vector store created by vector_db.py."""
    if not os.path.isdir(index_dir):
        raise FileNotFoundError(
            f"FAISS index '{index_dir}' not found. Run vector_db.py to build it first."
        )

    return FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def create_retriever(top_k: int = 5, **search_kwargs):
    """Create a langchain retriever backed by the FAISS vector store."""
    vector_store = load_vector_store()
    # Merge explicit kwargs with default top_k
    search_settings = {"k": top_k, **search_kwargs}
    return vector_store.as_retriever(search_kwargs=search_settings)


def retrieve_relevant_chunks(query: str, top_k: int = 5, **search_kwargs):
    """Return relevant document chunks for the provided query."""
    retriever = create_retriever(top_k=top_k, **search_kwargs)
    return retriever.invoke(query)


if __name__ == "__main__":
    query = input("Enter your query: ").strip()
    if not query:
        raise ValueError("Query cannot be empty.")

    top_k_input = input("Enter top_k (press Enter to use default 5): ").strip()
    top_k = int(top_k_input) if top_k_input else 5

    results = retrieve_relevant_chunks(query, top_k=top_k)
    if not results:
        print("No relevant chunks found for the given query.")
    else:
        print(f"Retrieved {len(results)} document chunks:")
        for idx, doc in enumerate(results, start=1):
            preview = doc.page_content.replace("\n", " ")
            print(f"{idx}. {preview[:200]}{'...' if len(preview) > 200 else ''}")
