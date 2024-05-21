import wandb
import requests
import streamlit as st
import weave
from langchain_core.messages import AIMessage, HumanMessage
import yaml


from src.RAG_pipeline.pipeline import create_pipeline


@weave.op() # logs input and output of calls 
def invoke_chain_helper_fn(chain, input, chat_history):
    
    chain_response =  chain.invoke(
        {
            "input": input,
            "chat_history": chat_history
        }
    )
    answer = chain_response["answer"]
    chat_history.extend([HumanMessage(content=input), AIMessage(content=answer)])

    return chain_response, answer, chat_history


def initialize_streamlit_session_states():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "contexts" not in st.session_state:
        st.session_state.contexts = []

    if "selected_context_index" not in st.session_state:
        st.session_state.selected_context_index = None

    if "Q_A" not in st.session_state:
        st.session_state.Q_A = []

    if "thumbs_feedback" not in st.session_state:
        st.session_state.thumbs_feedback = None
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def start_streamlit_app():
    st.title("Guide Helper")
    initialize_streamlit_session_states()


@weave.op()
def send_feedback(feedback):
    # Post the feedback to the Pipedream endpoint
    feedback_endpoint_response = requests.post(
        "https://eovyzv364l93k2e.m.pipedream.net", json=feedback
    )

    return feedback_endpoint_response


def feedback():
    
    # Thumbs up/down feedback button
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("👍"):
            st.session_state.thumbs_feedback = "good"
    with col2:
        if st.button("👎"):
            st.session_state.thumbs_feedback = "bad"

    # Feedback form
    with st.form(key="feedback_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        comment = st.text_area("Leave a comment")

        submitted = st.form_submit_button(label="Submit")

        if submitted:
            feedback = {
                "message_history": st.session_state.Q_A,
                "feedback": st.session_state.thumbs_feedback,
                "name": name,
                "email": email,
                "comment": comment,
                "url": [
                    chunk.metadata.get("url") for chunk in st.session_state.contexts
                ],
            }
            st.session_state.thumbs_feedback = None

            feedback_endpoint_response = send_feedback(feedback)

            if feedback_endpoint_response.status_code == 200:
                st.success("Feedback submitted successfully!")
            else:
                st.error("Failed to submit feedback.")





@weave.op()
def app(config: dict):

    chat_history = []
    retrieval_chain = create_pipeline(**config)


    start_streamlit_app() # title and session state variables

    # -------------------------------------------#


    # Start of query loop

    if input := st.chat_input("How can I assist you?"):
        st.session_state.messages.append({"role": "user", "content": input})

        st.chat_message("user").write(input)

        chain_response, answer, chat_history = invoke_chain_helper_fn(retrieval_chain, input, chat_history)
        context = chain_response["context"]

        #st.markdown(answer) # write ai response
        st.chat_message("assistant").write(answer)

        # Temporary Q&A list to store user input and assistant response which gets posted to Airtable
        # TODO This should be removed and handled in a better way TODO: Solved?
        st.session_state.Q_A = [
            {"role": "user", "content": input},
            {"role": "assistant", "content": answer},
        ]

        # Extract and store context in session state
        st.session_state.contexts = context
        st.session_state.selected_context_index = (
            0  # Reset index when new context is added
        )

        st.session_state.messages.append({"role": "assistant", "content": answer})

    # Check if the is a question and answer object in session state
    if st.session_state.Q_A:
        feedback()
        
    # Debugger: This create a drop down menu to show the retrieved context objects
    if st.session_state.contexts:
        # Created a selectbox to select the retrieved context objects
        selected_index = st.selectbox(
            "Context objects:",
            range(len(st.session_state.contexts)),
            format_func=lambda x: f"Retrieved context #{x + 1}",
            index=None,
        )

        # Update session state only if the selection changes
        if selected_index != st.session_state.selected_context_index:
            st.session_state.selected_context_index = selected_index

        # Only show context details if an item is selected
        if st.session_state.selected_context_index is not None:
            # Get the selected context object and display its metadata and content
            selected_context = st.session_state.contexts[
                st.session_state.selected_context_index
            ]
            st.json(body=selected_context.metadata)
            st.markdown(body=selected_context.page_content)



def load_config():
    with open("src/RAG_pipeline/conf/config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    return config


import os


if __name__ == "__main__":
    config = load_config()
    wandb.login(key=os.getenv("WEAVE_API_KEY"))
    weave.init(project_name="Guide Helper")
    app(config)

