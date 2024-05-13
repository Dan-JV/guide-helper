import os
import json
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError
from langchain.chains import Chain
from langchain.llms import OpenAI
from langchain.schema import Message, Role
from langchain.prompts import PromptPart
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

# Configuration for Qdrant
qdrant_endpoint = os.environ["Qdrant_endpoint"]
qdrant_api_key = os.environ["Qdrant_API_KEY"]

# Set up Qdrant client and vector store
client = QdrantClient(qdrant_endpoint, api_key=qdrant_api_key)
collection_name = "test"
embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
qdrant_client = Qdrant(client, collection_name, embeddings=embeddings_model)

# Create a retriever from Qdrant
retriever = qdrant_client.as_retriever()

# Initialize the Amazon Bedrock runtime client
boto_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

# Define function to invoke LLM using boto3
def invoke_llm(prompt):
    try:
        response = boto_client.invoke_model(
            modelId=model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ],
            })
        )
        result = json.loads(response.get("body").read())
        return result.get("content", [])[0].get("text", "")
    except ClientError as err:
        raise Exception(f"Error invoking LLM: {err}")

# Set up the LangChain chain
chain = Chain(
    components=[
        PromptPart("Enter your question: "),
        retriever,
        lambda docs, ctx: " ".join([doc.page_content for doc in docs]),
        invoke_llm,
    ],
    llm=OpenAI(api_key=os.environ["OPENAI_API_KEY"]),  # Setup your LLM key here
)

# Execute the chain
query = "Hvordan beregnes store bededagstillæg?"
result = chain.run(query)
print("Output:", result)
