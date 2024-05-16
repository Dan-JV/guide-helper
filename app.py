import streamlit as st
from dotenv import load_dotenv

from src.RAG_pipeline import pipeline


def app():

    retrieval_chain = pipeline.create_pipeline()

    st.title("Guide Helper")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

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

            response = st.write(messages["answer"])
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    load_dotenv()
    app() 