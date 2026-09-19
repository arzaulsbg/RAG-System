""" step-3: turn text into number (vectors) using jina"""

from langchain_community.embeddings import JinaEmbeddings

from hr_assistant import config

def get_embedding_model():
    """Return a Jina embeddings model. 
    Reads JINA_API_KEY from the environment.
    """
    return JinaEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)