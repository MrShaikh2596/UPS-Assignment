from typing import List

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from retriver import retrieve_relevant_chunks

try:
    from langchain_groq import ChatGroq
except ImportError as exc:
    raise ImportError(
        "langchain-groq must be installed and configured to generate LLM answers."
    ) from exc


env_path = find_dotenv()
if not env_path:
    raise FileNotFoundError(
        ".env file not found. Please ensure it exists in the project directory."
    )
print(f"Loading .env file from: {env_path}")
load_dotenv()

app = FastAPI(title="RAG Chat API")


llm = ChatGroq(model="openai/gpt-oss-120b")


class AnswerResponse(BaseModel):
    query: str
    top_k: int
    answer: str
    relevant_chunks: List[str]


def generate_answer(query: str, context_chunks: List[str]) -> str:
    """Use the configured LLM to synthesize an answer from the retrieved chunks."""
    if not context_chunks:
        return "No relevant information found to answer the question."

    system_prompt = (
        "You are a helpful assistant that answers questions based on the provided context. "
        "If the context does not contain the answer, say you do not know."
    )
    context = "\n\n".join(context_chunks)

    messages = [
        ("system", system_prompt),
        ("user", f"Context:\n{context}\n\nQuestion: {query}"),
    ]

    response = llm.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)


@app.get("/rag", response_model=AnswerResponse)
def rag_endpoint(query: str = Query(..., description="User question"), top_k: int = Query(5, ge=1, le=20)):
    try:
        docs = retrieve_relevant_chunks(query, top_k=top_k)
    except FileNotFoundError as missing_index:
        raise HTTPException(status_code=503, detail=str(missing_index)) from missing_index
    except Exception as err:  # pragma: no cover - defensive safety
        raise HTTPException(status_code=500, detail=str(err)) from err

    chunks = [doc.page_content for doc in docs]
    answer = generate_answer(query, chunks)

    payload = AnswerResponse(
        query=query,
        top_k=top_k,
        answer=answer,
        relevant_chunks=chunks,
    )
    return payload


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("rag-chat:app", host="0.0.0.0", port=8000, reload=False)
