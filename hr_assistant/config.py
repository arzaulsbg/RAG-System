"""All settings for the app live here, in one place"""

import os
from dotenv import load_dotenv

load_dotenv()

##Env variable
groq_api_key=os.getenv("GROQ_API_KEY")
jina_api_key=os.getenv("JINA_API_KEY")

##Data file path
DATA_FILE_PATH=os.path.join("data","hr_policy.txt")

##vector store

vector_STORE_PATH=os.path.join("data","faiss_index")

##LLM models and Embeddign model

LLM_MODEL_NAME="openai/gpt-oss-120b"

EMBEDDING_MODEL_NAME="jina-embeddings-v2-base-en"

## CHUNK AND TEXT SPLITTING CONFIG

CHUNK_SIZE=500
CHUNK_OVERLAP=50

## RETRIEVAL RESULT
TOP_K_RESULT=3

##SYSTEM INSTRUCTION


SYSTEM_PROMPT = (
    "You are a friendly HR assistant. Always use the search_hr_policy tool to look up "
    "facts before answering. If the answer isn't in the search results, say you don't know "
    "instead of guessing."
)

def check_api_keys() -> None:
    """Stop early with a clear message if a required API key is missing."""
    if not GROQ_API_KEY:
        raise ValueError("Missing GROQ_API_KEY. Please add it to your .env file.")
    if not JINA_API_KEY:
        raise ValueError("Missing JINA_API_KEY. Please add it to your .env file.")
   