import os

from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient


def initialize_qdrant_client(collection_name, embeddings_model):
    client = QdrantClient(
        os.getenv("QDRANT_ENDPOINT"), api_key=os.getenv("QDRANT_API_KEY")
    )
    qdrant_client = Qdrant(client, collection_name, embeddings=embeddings_model)

    return qdrant_client


def load_template(template_file_path):
    with open(template_file_path, 'r') as template_file:
        template_string = template_file.read()
    return template_string

