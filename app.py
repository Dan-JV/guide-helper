import requests
import streamlit as st
from streamlit import session_state as ss
import weave
import yaml

from langchain_core.runnables.history import RunnableWithMessageHistory

from src.RAG_pipeline.pipeline import create_pipeline

from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx


def _get_session_id():
    ctx = get_script_run_ctx()

    return ctx.session_id


# Streamlit config
st.set_page_config(
    page_title="guide buddy",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed",
    #    menu_items={
    #         'Get Help': 'https://www.extremelycoolapp.com/help',
    #         'Report a bug': "https://www.extremelycoolapp.com/bug",
    #         'About': "# This is a header. This is an *extremely* cool app!"
    #     }
)


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
        ss.model_choice = st.selectbox(
            "LLM",
            user_settings.get("models"),
            index=0,
            help="Select a model to generate responses",
        )

        # Response length choice
        ss.top_k = st.selectbox(
            "System response length",
            user_settings.get("response_length"),
            index=1,
            help="Sets the length of the generated responses by the system",
        )

        # Top K choice
        st.slider(
            "top_k",
            min_value=user_settings.get("top_k").get("min"),
            max_value=user_settings.get("top_k").get("max"),
            value=user_settings.get("top_k").get("default"),
            help="Number of chunks to retrieve from vector store",
        )

def get_output():
    return ss["output"]


@weave.op()
def logger(inputs):
    output = get_output()
    return output


def log_data(args):
    # construct outputs dict
    ss["output"] = args["output"]
    input = args["input"]
    logger_output = logger(input)

    return logger_output


def check_session_output_state():
    if ss["output"] is not None:
        log_data(
            
                {
                    "input": {
                        "chat_input": ss["chat_input"],
                        "system_prompt": ss["system_prompt"],
                        "config": ss.config,
                    },
                    "output": {
                        "answer": ss["output"]["answer"],
                        "context": ss["output"]["context"],
                        "feedback": ss["feedback"],
                    },
                }
        )
            
        

def set_it_up():
    
    st.title("Guide Buddy")
    
    if "config" not in ss:
        config = load_config()
        ss.config = config

    if "chain" not in ss:
        ss.chain: RunnableWithMessageHistory = create_pipeline(**ss.config["system_settings"])

    if "system_prompt" not in ss:
        system_prompts: list = ss.chain.get_prompts()
        ss.system_prompt = system_prompts[1].messages[0].prompt.template
        
    if "session_id" not in ss:
        ss.session_id = _get_session_id()

    if "messages" not in ss:
        ss.messages = []

    for message in ss.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if "contexts" not in ss:
        ss.contexts = []

    if "selected_context_index" not in ss:
        ss.selected_context_index = None

    if "Q_A" not in ss:
        ss.Q_A = []

    if "thumbs_feedback" not in ss:
        ss.thumbs_feedback = None

    # Debugger sidebar states
    if "model_choice" not in ss:
        ss.model_choice = config.get("model_id")

    if "response_length" not in ss:
        ss.response_length = "normal"

    if "top_k" not in ss:
        ss.top_k = config.get("top_k")

    if "output" not in ss:
        ss["output"] = None
    
    if "logged_previous_message" not in ss:
        ss["logged_previous_message"] = False

    if "previous_message" not in ss:
        ss["previous_message"] = None
    
    if "submitted" not in ss:
        ss["submitted"] = False
    
    
    run_app()
    

def submitted_callback():
    ss.submitted = True

def run_app():
    # Debugger sidebar
    # debugger_sidebar(config.get("user_settings"))

    if chat_input := st.chat_input(
        "How can I assist you?"
    ):
        if ss.previous_message is None:
            ss.previous_message = chat_input
        else:
            if not ss["logged_previous_message"]:
                input = {
                            "input": {
                                "chat_input": ss.previous_message,
                                "system_prompt": ss.system_prompt,
                                "chat_history": ss.chain.get_session_history(ss.session_id),
                                "config": ss.config,
                            },
                            "output": {
                                "answer": ss["output"]["answer"],
                                "context": ss["output"]["context"],
                                "feedback": ss["feedback"],
                            },
                        }
                    
                
                log_data(input)
                ss.logged_previous_message = True
        st.chat_message("user").write(chat_input)
        ss["chat_input"] = chat_input
        ss.messages.append({"role": "user", "content": chat_input})

        chat_history = ss.chain.get_session_history(ss.session_id)

        ss.output = ss.chain.invoke(
            {"input": chat_input, "chat_history": chat_history},
            config={"configurable": {"session_id": ss.session_id}},
        )
        st.chat_message("assistant").write(ss.output["answer"])
        ss.messages.append(
            {"role": "assistant", "content": ss.output["answer"]}
        )
        ss.logged_previous_message = False


        ss["Q_A"] = [
            {"role": "user", "content": ss["chat_input"]},
            {"role": "guide_helper", "content": ss["output"]["answer"]},
        ]
        
        context = ss.output["context"]
        ss.contexts = context
        ss.selected_context_index = (
            0  # Reset index when new context is added
        )


    # Feedback form
    if ss.Q_A:
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("👍"):
                ss.thumbs_feedback = "good"
        with col2:
            if st.button("👎"):
                ss.thumbs_feedback = "bad"
        with st.form(key="feedback_form", clear_on_submit=False):
            name = st.text_input("Name")
            email = st.text_input("Email")
            comment = st.text_area("Leave a comment")

            feedback = {
                "message_history": ss["Q_A"],
                "name": name,
                "email": email,
                "comment": comment,
                "feedback": ss.thumbs_feedback,
                "url": [
                    chunk.metadata.get("url")
                    for chunk in ss["output"]["context"]
                ],
            }
            ss["feedback"] = feedback

            st.form_submit_button(
                "Submit Feedback",
                on_click=submitted_callback
            )

            if ss.submitted: 
                ss["logged_previous_message"] = True
                ss.previous_message = chat_input
                input = {
                            "input": {
                                "chat_input": chat_input,
                                "system_prompt": ss.system_prompt,
                                "chat_history": ss.chain.get_session_history(ss.session_id),
                                "config": ss.config,
                            },
                            "output": {
                                "answer": ss["output"]["answer"],
                                "context": ss["output"]["context"],
                                "feedback": ss["feedback"],
                            },
                        } 
                log_data(input)

                ss.thumbs_feedback = None

                feedback_endpoint_response = send_feedback(ss["feedback"])

                if feedback_endpoint_response.status_code == 200:
                    st.success("Thank you for your feedback!")
                else:
                    st.error("Failed to submit feedback.")
                
                ss.Q_A = []
                ss.submitted = False


        if ss.contexts:
            # Created a selectbox to select the retrieved context objects
            selected_index = st.selectbox(
                "Context objects:",
                range(len(ss.contexts)),
                format_func=lambda x: f"Retrieved context #{x + 1}",
                index=None,
            )

            # Update session state only if the selection changes
            if selected_index != ss.selected_context_index:
                ss.selected_context_index = selected_index

            # Only show context details if an item is selected
            if ss.selected_context_index is not None:
                # Get the selected context object and display its metadata and content
                selected_context = ss.contexts[
                    ss.selected_context_index
                ]
                st.json(body=selected_context.metadata)
                st.markdown(body=selected_context.page_content)


def load_config():
    with open("src/RAG_pipeline/conf/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    weave.init(project_name="Guide Helper")
    set_it_up()
