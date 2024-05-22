import requests
import streamlit as st
import weave
from langchain_core.messages import AIMessage, HumanMessage
import yaml

from langchain_core.runnables.history import RunnableWithMessageHistory

from src.RAG_pipeline.pipeline import create_pipeline

from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx


def _get_session_id():
    ctx = get_script_run_ctx()

    return ctx.session_id

# Streamlit config
st.set_page_config(page_title= "guide buddy",
                   page_icon=None, 
                   layout="centered", 
                   initial_sidebar_state="collapsed", 
                #    menu_items={
                #         'Get Help': 'https://www.extremelycoolapp.com/help',
                #         'Report a bug': "https://www.extremelycoolapp.com/bug",
                #         'About': "# This is a header. This is an *extremely* cool app!"
                #     }
                    )

class GuideHelper:
    def __init__(self, config: dict):
        self.config = config
        self.user_settings = self.config.get("user_settings")

        st.title("Guide Helper")

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

        if "session_id" not in st.session_state:
            st.session_state.session_id = _get_session_id()


        # Debugger sidebar states
        if "model_choice" not in st.session_state:
            st.session_state.model_choice = config.get("model_id")

        if "response_length" not in st.session_state:
            st.session_state.response_length = "normal"

        if "top_k" not in st.session_state:
            st.session_state.top_k = config.get("top_k")
        
        self.chain: RunnableWithMessageHistory = create_pipeline(**config["system_settings"])
        self.system_prompts: list = self.chain.get_prompts()
        self.system_prompt: str = self.system_prompts[1].messages[0].prompt.template
    

    @weave.op()
    def on_chat_submit(self, user_input, system_prompt, **config):
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        chain_response = self.chain.invoke(
            {
                "input": user_input,
                "chat_history": self.chain.get_session_history(st.session_state.session_id)
            },
            config={
                "configurable":
                {
                    "session_id": st.session_state.session_id
                }
            }
        )
        answer = chain_response["answer"]
        context = chain_response["context"]

        st.chat_message("assistant").write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        
        # for the feedback part
        question_answer = [
            {"role": "user", "content": user_input},
            {"role": "guide_helper", "content": answer},
        ]

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
                    "message_history": question_answer,
                    "feedback": st.session_state.thumbs_feedback,
                    "name": name,
                    "email": email,
                    "comment": comment,
                    "url": [
                        chunk.metadata.get("url") for chunk in context
                    ],
                }
                st.session_state.thumbs_feedback = None

                feedback_endpoint_response = send_feedback(feedback)

                if feedback_endpoint_response.status_code == 200:
                    st.success("Feedback submitted successfully!")
                else:
                    st.error("Failed to submit feedback.")
            else:
                feedback = None

        
        
        output = {
            "answer": answer,
            "context": context,
        }
        if feedback is not None:
            output.update(feedback)
        else:
            feedback = {"feedback": None}
            output.update(feedback)
        
        return output

        
def send_feedback(feedback):
    # Post the feedback to the Pipedream endpoint
    feedback_endpoint_response = requests.post(
        "https://eovyzv364l93k2e.m.pipedream.net", json=feedback
    )

    return feedback_endpoint_response       


def debugger_sidebar(user_settings: dict) -> None:
    with st.sidebar:
        st.title("Guide Helper - Debug menu")
        
        # Model choice
        st.session_state.model_choice = st.selectbox("LLM", 
                     user_settings.get("models"),
                     index=0,
                     help="Select a model to generate responses")
        
        # Response length choice
        st.session_state.top_k = st.selectbox("System response length", 
                     user_settings.get("response_length"),
                     index=1,
                     help="Sets the length of the generated responses by the system")
        
        # Top K choice
        st.slider("top_k", 
                  min_value=user_settings.get("top_k").get("min"),
                  max_value=user_settings.get("top_k").get("max"),
                  value=user_settings.get("top_k").get("default"),
                  help="Number of chunks to retrieve from vector store",)

def run_app(config: dict):
    guide_helper = GuideHelper(config)
    system_prompt = guide_helper.system_prompt

    # Debugger sidebar
    debugger_sidebar(guide_helper.user_settings)

    chat_input = st.chat_input("How can I assist you?")

    if chat_input:
        output = guide_helper.on_chat_submit(chat_input, system_prompt, **config)
        context = output["context"]

        st.session_state.contexts = context
        st.session_state.selected_context_index = (
            0  # Reset index when new context is added
        )

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
    with open("src/RAG_pipeline/conf/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    config = load_config()
    weave.init(project_name="Guide Helper")
    run_app(config)
