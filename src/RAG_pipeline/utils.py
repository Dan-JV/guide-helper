import os
from typing import Any

from jinja2 import Environment
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from jinja2 import Environment, FileSystemLoader, select_autoescape


def initialize_qdrant_client(collection_name: str, embeddings_model: BedrockEmbeddings) -> Qdrant:
    client = QdrantClient(
        os.getenv("QDRANT_ENDPOINT"), api_key=os.getenv("QDRANT_API_KEY")
    )
    qdrant_client = Qdrant(client, collection_name, embeddings=embeddings_model)

    return qdrant_client


def load_template(template_file_path):
    with open(template_file_path, "r") as template_file:
        template_string = template_file.read()
    return template_string


def render_jinja_template(template_file: str, data: dict[str, Any]) -> str:
    jinja_env = Environment(
        loader=FileSystemLoader(searchpath="./"), autoescape=select_autoescape()
    )
    template = jinja_env.get_template(template_file)
    return template.render(**data)


def fill_in_template(
    template_file_path, system_instructions_path, fewshot_examples_path
):
    template = load_template(template_file_path)

    system_instructions = load_template(system_instructions_path)

    fewshot_examples = load_template(fewshot_examples_path)

    template = render_jinja_template(
        template_file_path,
        {
            "system_instructions": system_instructions,
            "fewshot_examples": fewshot_examples,
        },
    )
    template += "{context}"

    return template
