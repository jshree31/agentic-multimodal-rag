import streamlit as st
import requests
import os
import json

# ---------------------------
# API CONFIG
# ---------------------------
CHAT_API_URL = "http://127.0.0.1:8000/api/v1/query"
UPLOAD_API_URL = "http://127.0.0.1:8000/api/v1/upload"

# ---------------------------
# FUNCTION: SEND QUERY
# ---------------------------
def send_query(query):
    try:
        response = requests.post(
            CHAT_API_URL,
            json={
                "query": query
                #"insurance_data": insurance_data
            }
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="AI Insurance Assistant",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------
# SIDEBAR NAVIGATION
# ---------------------------
with st.sidebar:
    st.title("AI Assistant")

    page = st.radio(
        "Navigation",
        ["Chat", "Admin"]
    )

    st.divider()

# =========================================================
# 💬 CHAT PAGE
# =========================================================
if page == "Chat":

    st.title("Smart Bank Assistant")
 
    # ---------------------------
    # SESSION STATE
    # ---------------------------
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}

    if "current_chat" not in st.session_state:
        st.session_state.current_chat = "Chat 1"

    if "Chat 1" not in st.session_state.chat_sessions:
        st.session_state.chat_sessions["Chat 1"] = []

    # ---------------------------
    # CHAT SIDEBAR
    # ---------------------------
    with st.sidebar:
        st.markdown("### 💬 Chats")

        if st.button("New Chat"):
            new_chat = f"Chat {len(st.session_state.chat_sessions) + 1}"
            st.session_state.chat_sessions[new_chat] = []
            st.session_state.current_chat = new_chat

        for chat in st.session_state.chat_sessions:
            if st.button(f"📁 {chat}"):
                st.session_state.current_chat = chat

    # ---------------------------
    # LOAD CHAT
    # ---------------------------
    current_chat = st.session_state.current_chat
    messages = st.session_state.chat_sessions[current_chat]

    st.markdown(f"### 💬 {current_chat}")
    st.divider()

    # ---------------------------
    # DISPLAY MESSAGES
    # ---------------------------
    if len(messages) == 0:
        st.info("👋 Start a conversation...")

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ---------------------------
    # USER INPUT
    # ---------------------------
    user_input = st.chat_input("Enter loan details or ask a question...")

    if user_input:
        # Save user message
        messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        # API CALL
        with st.chat_message("assistant"):
            with st.spinner("Thinking... 🤔"):

                response_json = send_query(user_input)

                if "error" in response_json:
                    st.error(response_json["error"])
                    response_text = response_json["error"]
                else:
                    main_text = response_json.get("response", "")
                    page_no = response_json.get("page")
                    doc_name = response_json.get("doc_name")
                    confidence = response_json.get("confidence")

                    meta = f"\n\n📄 Page: {page_no} | Doc: {doc_name}" if page_no else ""
                    conf = f"\n🎯 Confidence: {confidence}" if confidence else ""

                    response_text = f"{main_text}{meta}{conf}"

                    st.markdown(response_text)

        messages.append({"role": "assistant", "content": response_text})

# =========================================================
# 🛠️ ADMIN PAGE (UNCHANGED)
# =========================================================
elif page == "Admin":

    st.title("Admin Dashboard")

    password = st.text_input("Enter Admin Password", type="password")

    if password != "admin123":
        st.warning("🔒 Admin access only")
        st.stop()

    st.success("✅ Access Granted")

    st.markdown("### 📄 Upload Bank Documents")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        if st.button("🚀 Upload"):
            files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            response = requests.post(UPLOAD_API_URL, files=files)

            if response.status_code == 200:
                st.success("✅ File uploaded successfully!")
            else:
                st.error("❌ Upload failed!")