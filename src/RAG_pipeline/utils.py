import os
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient


def initialize_qdrant_client(collection_name, embeddings_model):
    client = QdrantClient(
        os.getenv("QDRANT_ENDPOINT"), api_key=os.getenv("QDRANT_API_KEY")
    )
    qdrant_client = Qdrant(client, collection_name, embeddings=embeddings_model)

    return qdrant_client


def process_template(template_file: str, data: dict[str, Any]) -> str:
    """Process a Jinja template file with the provided data.
    Example:
    ```
    process_template("prompts/guide_helper.jinja", {"context": "context", "input": "input"})
    ```
    """
    jinja_env = Environment(
        loader=FileSystemLoader(searchpath="./"), autoescape=select_autoescape()
    )
    template = jinja_env.get_template(template_file)
    return template.render(**data)

