"""Step 4 : store chunk embedding in FAISS sowe can search them later"""

import os
from langchain_community.vectorstores import FAISS

from hr_assistant import config

from hr_assistant.embeddings import get_embedding_model

# build vector_store that is save during session

def build_vector_store(chunks):

    """
    Embed evrery chunk and build a searchable FAISS index in memory.
    """
    embedding_model=get_embedding_model()
    return FAISS.from_documents(chunks,embedding_model)

##save vectoe store

def save_vector_store(vector_store,path:str=config.vector_STORE_PATH)->None:
    """
    Save the FAISS index to disk
    so we don't have to rebuild it every time.
    """
    vector_store.save_local(path)
## load vectore store
def load_vector_store(path:str=config.vector_STORE_PATH):
    """
    Load a previously saved FAISS index from dist"""
    embedding_model=get_embedding_model()
    ##allow_dangerous_deserialization is safe here because we only ever load
    ## an index that this same app created and aved
    return FAISS.load_local(path,embedding_model,allow_dangerous_deserialization=True)

## check vector_store exist or not

def vector_store_exists(path:str=config.vector_STORE_PATH)->bool:
    """check if a saved FAISS index already exist on disk"""
    return os.path.exists(os.path.join(path,"index.faiss"))


##Retrieve data

def get_retriever(vector_store,k:int=config.TOP_K_RESULT):
    """Turn a vector store into a retriever that return the top-k matching chunks."""
    return vector_store.as_retriever(search_kwargs={"k":k})