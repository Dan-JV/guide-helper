import os

import boto3
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_aws import BedrockChat
from langchain_community.embeddings import BedrockEmbeddings
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts.chat import ChatPromptTemplate

from langchain.chains import create_history_aware_retriever

from src.RAG_pipeline.utils import initialize_qdrant_client, fill_in_template

def create_pipeline(model_id: str, region_name: str, embedding_model_id: str, collection_name: str, search_kwargs: dict):

    bedrock_runtime_client = boto3.client(
        service_name="bedrock-runtime",
        region_name=region_name,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    embedding_model_id = embedding_model_id
    embeddings_model = BedrockEmbeddings(
        client=bedrock_runtime_client, model_id=embedding_model_id
    )

    qdrant_client = initialize_qdrant_client(collection_name, embeddings_model)

    retriever = qdrant_client.as_retriever(search_kwargs=search_kwargs)

    # Promt construction
    prompt_path = "prompts/guide_helper.jinja"
    system_prompt_path = "prompts/system_instructions.jinja"
    fewshot_examples_path = "prompts/fewshot_prompt_questions_answers.jinja"
    prompt_template = fill_in_template(prompt_path, system_prompt_path, fewshot_examples_path)


    llm = BedrockChat(
                client=bedrock_runtime_client,
                credentials_profile_name="default",
                model_id=model_id,
                streaming=True
            )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


