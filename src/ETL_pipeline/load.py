import os

import boto3
from dotenv import load_dotenv
from langchain.vectorstores import Qdrant
from langchain_community.embeddings import BedrockEmbeddings
from qdrant_client import QdrantClient, models


def load(docs):
    load_dotenv()

    qdrant_endpoint = os.environ["Qdrant_endpoint"]
    qdrant_api_key = os.environ["Qdrant_API_KEY"]

    qdrant_client = QdrantClient(qdrant_endpoint, api_key=qdrant_api_key)

    session = boto3.Session()
    bedrock_client = session.client(
        "bedrock-runtime",
        region_name="us-east-1",
    )

    collection_name = "markdown_header_level2_1024split_64overlap"

    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name)
    qdrant_client.create_collection(
        f"{collection_name}",
        vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
    )

    embeddings = BedrockEmbeddings(
        client=bedrock_client, model_id="amazon.titan-embed-text-v2:0"
    )

    BATCH_SIZE = 16
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]

        doc_store = Qdrant.from_documents(
            batch,
            embeddings,
            url=qdrant_endpoint,
            api_key=qdrant_api_key,
            prefer_grpc=True,
            collection_name=collection_name,
        )
