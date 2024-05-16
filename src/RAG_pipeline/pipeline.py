
import boto3
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_aws import BedrockChat
from langchain_community.embeddings import BedrockEmbeddings
from langchain_core.prompts import PromptTemplate

from src.RAG_pipeline.utils import initialize_qdrant_client


def create_pipeline():
    model_id = "anthropic.claude-instant-v1"
    bedrock_runtime_client = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
    )

    embedding_model_id = "amazon.titan-embed-text-v1"
    embeddings_model = BedrockEmbeddings(
        client=bedrock_runtime_client, model_id=embedding_model_id
    )

    collection_name = "text-titan-embed-text-v1"
    qdrant_client = initialize_qdrant_client(collection_name, embeddings_model)

    retriever = qdrant_client.as_retriever(
        search_type="mmr", search_kwargs={"k": 4, "fetch_k": 30}
    )
    prompt = PromptTemplate.from_template(
        "As an expert system designed to assist with inquiries about Visma Enterprise A/S's internal guides, you are to provide answers that are strictly based on the content of these guides.\nUse specific terminology and keywords from the guides to maintain consistency with Visma’s standards.\nMaintain a professional tone throughout your response and ensure that the information is both comprehensive and precise.\nIf applicable, reference specific sections or points within the guides to add clarity and relevance to your answer.\nAll responses must be provided in Danish.\nIf a query is vague or you do not have sufficient information to provide an accurate response, ask for clarification or admit that you do not know the answer.\nAvoid providing speculative information or details outside the scope of the internal guides.\n\n<context>\n{context}\n</context>\n\nUser question:{input}"
    )

    llm = BedrockChat(
                client=bedrock_runtime_client,
                credentials_profile_name="default",
                model_id=model_id,
                streaming=True
            )

    combine_docs_chain = create_stuff_documents_chain(
        llm, prompt
    )
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return retrieval_chain
