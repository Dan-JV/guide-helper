import os
import requests
import streamlit as st
from dotenv import load_dotenv

from src.RAG_pipeline import pipeline


def app():

    os.environ["QDRANT_API_KEY"] = st.secrets["QDRANT_API_KEY"]
    os.environ["QDRANT_ENDPOINT"] = st.secrets["QDRANT_ENDPOINT"]
    os.environ["AWS_ACCESS_KEY_ID"] = st.secrets["AWS_ACCESS_KEY_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["AWS_SECRET_ACCESS_KEY"]

    retrieval_chain = pipeline.create_pipeline()

    st.title("Guide Helper")

    # Session state variables

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

    # -------------------------------------------#

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Start of query loop

    if input := st.chat_input("How can I assist you?"):
        st.session_state.messages.append({"role": "user", "content": input})

        with st.chat_message("user"):
            st.markdown(input)

        with st.chat_message("assistant"):
            
            chain_output = retrieval_chain.invoke(
                {
                    "input": input,
                }
            )

            response = chain_output["answer"]
            st.write(response)

            # Temporary Q&A list to store user input and assistant response which gets posted to Airtable
            # TODO This should be removed and handled in a better way
            st.session_state.Q_A = [{"role": "user", "content": input},{"role": "assistant", "content": response}]

            # Extract and store context in session state
            st.session_state.contexts = chain_output.get("context", [])
            st.session_state.selected_context_index = 0  # Reset index when new context is added

        st.session_state.messages.append({"role": "assistant", "content": response})

    # Check if the is a question and answer object in session state
    if st.session_state.Q_A:

        # Thumbs up/down feedback button
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button('👍'):
                st.session_state.thumbs_feedback = 'good'
        with col2:
            if st.button('👎'):
                st.session_state.thumbs_feedback = 'bad'

        # Feedback form
        with st.form(key='feedback_form'):

            # Populate the feedback form
            name = st.text_input("Name")
            email = st.text_input("Email")
            comment = st.text_area("Leave a comment")

            # Submit button
            submitted = st.form_submit_button(label='Submit')

            if submitted:
                # Feedback form dict
                feedback ={
                    'message_history': st.session_state.Q_A,
                    'feedback': st.session_state.thumbs_feedback,
                    'name' : name, 
                    'email' : email,
                    'comment': comment,
                    'url' : [chunk.metadata.get("url") for chunk in st.session_state.contexts]
                }
                st.session_state.thumbs_feedback = None

                # Post the feedback to the Pipedream endpoint
                feedback_endpoint_response = requests.post("https://eovyzv364l93k2e.m.pipedream.net", json=feedback)
                if feedback_endpoint_response.status_code == 200:
                    st.success("Feedback submitted successfully!")
                else:
                    st.error("Failed to submit feedback.")

    # Debugger: This create a drop down menu to show the retrieved context objects
    if st.session_state.contexts:
        # Created a selectbox to select the retrieved context objects
        selected_index = st.selectbox(
            "Context objects:",
            range(len(st.session_state.contexts)),
            format_func=lambda x: f"Retrieved context #{x + 1}",
            index=None
        )

        # Update session state only if the selection changes
        if selected_index != st.session_state.selected_context_index:
            st.session_state.selected_context_index = selected_index

        # Only show context details if an item is selected
        if st.session_state.selected_context_index is not None:
            # Get the selected context object and display its metadata and content
            selected_context = st.session_state.contexts[st.session_state.selected_context_index]
            st.json(body=selected_context.metadata)
            st.markdown(body=selected_context.page_content)

if __name__ == '__main__':
    load_dotenv()
    app()