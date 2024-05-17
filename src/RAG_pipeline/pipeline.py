import os

import boto3
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_aws import BedrockChat
from langchain_community.embeddings import BedrockEmbeddings
from langchain_core.prompts import PromptTemplate

from src.RAG_pipeline.utils import initialize_qdrant_client, load_template


def create_pipeline():

    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    bedrock_runtime_client = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    embedding_model_id = "amazon.titan-embed-text-v1"
    embeddings_model = BedrockEmbeddings(
        client=bedrock_runtime_client, model_id=embedding_model_id
    )

    collection_name = "text-titan-embed-text-v1"
    qdrant_client = initialize_qdrant_client(collection_name, embeddings_model)

    # Retriever 
    retriever = qdrant_client.as_retriever(search_kwargs={'k': 10})

    # Promt construction
    prompt_template_string = load_template("prompts/guide_helper.jinja") 
    few_shot_examples_template_string = load_template("prompts/fewshot_prompt_questions_answers.jinja")
    prompt_template = PromptTemplate.from_template(
        prompt_template_string, template_format="jinja2", partial_variables={"few_shot_examples": few_shot_examples_template_string}
    )
    prompt_template = prompt_template.partial(few_shot_examples=few_shot_examples_template_string)

    llm = BedrockChat(
                client=bedrock_runtime_client,
                credentials_profile_name="default",
                model_id=model_id,
                streaming=True
            )

    combine_docs_chain = create_stuff_documents_chain(
        llm, prompt_template
    )
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return retrieval_chain

