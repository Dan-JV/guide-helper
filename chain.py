import os
import json
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError
from langchain_aws import BedrockLLM
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import BedrockChat
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("As an expert system designed to assist with inquiries about Visma Enterprise A/S's internal guides, you are to provide answers that are strictly based on the content of these guides.\nUse specific terminology and keywords from the guides to maintain consistency with Visma’s standards.\nMaintain a professional tone throughout your response and ensure that the information is both comprehensive and precise.\nIf applicable, reference specific sections or points within the guides to add clarity and relevance to your answer.\nAll responses must be provided in Danish.\nIf a query is vague or you do not have sufficient information to provide an accurate response, ask for clarification or admit that you do not know the answer.\nAvoid providing speculative information or details outside the scope of the internal guides.\n\n<context>\n{context}\n</context>\n\nUser question:{input}")

from qdrant_client import QdrantClient

from helperfunctions.RAG_prompt import RAGPrompt
from helperfunctions.format_docs import format_docs

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
retriever = qdrant_client.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 1, 'fetch_k': 20}
            )

# Initialize the Amazon Bedrock runtime client
boto_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

llm = BedrockChat(
                credentials_profile_name="default",
                model_id=model_id,
                streaming=True
            )

combine_docs_chain = create_stuff_documents_chain(
    llm, prompt
)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)


results = retrieval_chain.invoke({"input":"Hvilke rapporter beskriver bededagstillæg?"})

print(results)