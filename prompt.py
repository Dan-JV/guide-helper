import os

from botocore.exceptions import ClientError
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

load_dotenv()


qdrant_endpoint = os.environ["Qdrant_endpoint"]
qdrant_api_key = os.environ["Qdrant_API_KEY"]

client = QdrantClient(
    qdrant_endpoint,
    api_key=qdrant_api_key
)
collection_name = "test"
qdrant_client = Qdrant(
    client,
    collection_name,
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3"
    )
)

retriever = qdrant_client.as_retriever()
query = "Hvordan beregnes store bededagstillæg?"
retrieved_docs = retriever.invoke(query)
retrieved_docs = [doc.page_content for doc in retrieved_docs]
retrieved_docs = "".join(retrieved_docs)

import json
import logging

import boto3

logger = logging.getLogger(__name__)

def invoke_claude_3_with_text(prompt):
    """
    Invokes Anthropic Claude 3 Sonnet to run an inference using the input
    provided in the request body.

    :param prompt: The prompt that you want Claude 3 to complete.
    :return: Inference response from the model.
    """

    # Initialize the Amazon Bedrock runtime client
    client =  boto3.client(
        service_name="bedrock-runtime", region_name="us-east-1"
    )

    # Invoke Claude 3 with the text prompt
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

    try:
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1024,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}],
                        }
                    ],
                }
            ),
        )

        # Process and print the response
        result = json.loads(response.get("body").read())
        input_tokens = result["usage"]["input_tokens"]
        output_tokens = result["usage"]["output_tokens"]
        output_list = result.get("content", [])

        print("Invocation details:")
        print(f"- The input length is {input_tokens} tokens.")
        print(f"- The output length is {output_tokens} tokens.")

        print(f"- The model returned {len(output_list)} response(s):")
        for output in output_list:
            print(output["text"])

        return result

    except ClientError as err:
        logger.error(
            "Couldn't invoke Claude 3 Sonnet. Here's why: %s: %s",
            err.response["Error"]["Code"],
            err.response["Error"]["Message"],
        )
        raise


prompt = f"{retrieved_docs} {query}"

result = invoke_claude_3_with_text(prompt)
print(result)