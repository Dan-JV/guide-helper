import json
import os

import boto3
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain_core.documents.base import Document
from llama_index.core import SimpleDirectoryReader
from qdrant_client import QdrantClient, models
from langchain_community.embeddings import BedrockEmbeddings

load_dotenv()

qdrant_endpoint = os.environ["Qdrant_endpoint"]
qdrant_api_key = os.environ["Qdrant_API_KEY"]

qdrant_client = QdrantClient(
    qdrant_endpoint,
    api_key=qdrant_api_key
)

session = boto3.Session()
bedrock_client = session.client(
    "bedrock-runtime",
    region_name="us-east-1",
)

collection_name = "text-titan-embed-text-v1"
if qdrant_client.collection_exists(collection_name):
    qdrant_client.delete_collection(collection_name)
qdrant_client.create_collection(
    f"{collection_name}",
    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
)

# embeddings = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-m3"
# )
embeddings = BedrockEmbeddings(client=bedrock_client, model_id="amazon.titan-embed-text-v1") 

def chunks_helper():
    reader = SimpleDirectoryReader(input_dir="data/chunks", recursive=True)
    docs = list()
    for chunk in reader.iter_data():
        chunk = chunk[0]
        text = chunk.text
        text_dict = json.loads(text)
        metadata = text_dict['metadata']
        page_content = text_dict['content']
        doc = Document(metadata=metadata, page_content=page_content)
        docs.append(doc)

    return docs
docs = chunks_helper()

BATCH_SIZE=4
for i in range(0, len(docs), BATCH_SIZE):
    batch = docs[i:i+BATCH_SIZE]

    doc_store = Qdrant.from_documents(
        batch, embeddings, url=qdrant_endpoint, api_key=qdrant_api_key, prefer_grpc=True, collection_name=collection_name)
