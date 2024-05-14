import json
import logging
import os

import boto3
import streamlit as st
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from openai import OpenAI
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

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


def process_retrived_docs(retrieved_docs):
    retrieved_docs = [doc.page_content for doc in retrieved_docs]
    retrieved_docs = "".join(retrieved_docs)
    return retrieved_docs

def retrieve_docs(query):
    retrieved_docs = retriever.invoke(query)
    retrieved_docs = process_retrived_docs(retrieved_docs)
    return retrieved_docs


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




st.title("ChatGPT-like clone")

api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    retrieved_docs = retrieve_docs(prompt)
    prompt = f"{retrieved_docs} {prompt}"
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})