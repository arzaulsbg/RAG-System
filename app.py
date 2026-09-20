import streamlit as st
from hr_assistant.pipeline import ask, build_hr_assistant


# Page configuration
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="🤖",
    layout="centered"
)


# Build assistant once and reuse it
@st.cache_resource
def load_assistant():
    return build_hr_assistant()


# Load assistant
with st.spinner("Loading HR Policy Assistant..."):
    agent = load_assistant()


# Header
st.title("🤖 HR Policy Assistant")
st.write("Ask questions about company HR policies.")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
question = st.chat_input("Ask a question about HR policies...")


if question:
    # Display user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            try:
                answer = ask(agent, question)
                st.markdown(answer)

            except Exception as e:
                answer = f"Sorry, something went wrong: {e}"
                st.error(answer)

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })


# Sidebar
with st.sidebar:
    st.header("🤖 HR Policy Assistant")

    st.write(
        "This assistant answers questions using the "
        "company's HR policy documents."
    )

    st.divider()

    st.subheader("Example Questions")

    examples = [
        "How many paid annual leave days do I get?",
        "What is the notice period during probation?",
        "Can I work from home every day?"
    ]

    for example in examples:
        st.write(f"• {example}")

    st.divider()

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()