import os

from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient


def initialize_qdrant_client(collection_name, embeddings_model):
    qdrant_endpoint = os.environ["Qdrant_endpoint"]
    qdrant_api_key = os.environ["Qdrant_API_KEY"]
    client = QdrantClient(qdrant_endpoint, api_key=qdrant_api_key)
    qdrant_client = Qdrant(client, collection_name, embeddings=embeddings_model)


    return qdrant_client