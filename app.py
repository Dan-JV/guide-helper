import os

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

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Start of query loop

    if prompt := st.chat_input("How can I assist you?"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            messages_string = "\n".join(
                [f"{m['role']}: {m['content']}" for m in st.session_state.messages]
            )
            messages = retrieval_chain.invoke(
                {
                    "input": messages_string,
                }
            )

            response = messages["answer"]
            st.write(response)

            # Extract and store context in session state
            st.session_state.contexts = messages.get("context", [])
            st.session_state.selected_context_index = 0  # Reset index when new context is added

        st.session_state.messages.append({"role": "assistant", "content": response})

    
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