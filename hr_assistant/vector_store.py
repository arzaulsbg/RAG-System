"""Step 4 : store chunk embedding in FAISS sowe can search them later"""

import os
# from langchain_community.vectorstores import FAISS

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from hr_assistant import config

from hr_assistant.embeddings import get_embedding_model


from hr_assistant.logger import get_logger

logger = get_logger(__name__)



# build vector_store that is save during session

def build_vector_store(chunks):
    """Embed every chunk and upload it into a Qdrant Cloud collection."""
    logger.info(
        "Embedding %d chunk(s) and uploading to Qdrant collection '%s'...",
        len(chunks),
        config.QDRANT_COLLECTION_NAME,
    )
    embeddings_model = get_embedding_model()
    vector_store = QdrantVectorStore.from_documents(
        chunks,
        embedding=embeddings_model,
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        collection_name=config.QDRANT_COLLECTION_NAME,
    )
    logger.info("Uploaded to Qdrant collection '%s'", config.QDRANT_COLLECTION_NAME)
    return vector_store

# ##save vectoe store

# def save_vector_store(vector_store,path:str=config.vector_STORE_PATH)->None:
#     """
#     Save the FAISS index to disk
#     so we don't have to rebuild it every time.
#     """
#     vector_store.save_local(path)
#     logger.info("Saved FAISS index to '%s'",path)



## load vectore store

def load_vector_store():
    """Connect to a Qdrant Cloud collection that was already built before."""
    logger.info("Connecting to existing Qdrant collection '%s'", config.QDRANT_COLLECTION_NAME)
    embeddings_model = get_embedding_model()
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings_model,
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        collection_name=config.QDRANT_COLLECTION_NAME,
    )

## check vector_store exist or not

def vector_store_exists() -> bool:
    """Check if the Qdrant Cloud collection already exists."""
    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    return client.collection_exists(config.QDRANT_COLLECTION_NAME)


##Retrieve data

def get_retriever(vector_store,k:int=config.TOP_K_RESULT):
    """Turn a vector store into a retriever that return the top-k matching chunks."""
    logger.info("Creating retriever with top_k=%d",k)
    return vector_store.as_retriever(search_kwargs={"k":k})