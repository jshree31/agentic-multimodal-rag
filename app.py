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
                    # ✅ NEW ROUTE-AWARE RENDERING
                    route = response_json.get("route")

                    st.markdown(f"**🔀 Route Used:** {route}")
                    st.markdown(response_json.get("answer", ""))

                    # ─────────────────────────────
                    # BANKING RESPONSE
                    # ─────────────────────────────
                    if route == "banking":
                        st.subheader("🧾 SQL Query Executed")
                        st.code(response_json.get("sql_query_executed", ""), language="sql")

                        st.subheader("📊 SQL Result")
                        sql_result = response_json.get("sql_result")

                        if isinstance(sql_result, list):
                            st.json(sql_result)
                        else:
                            st.write(sql_result)

                        st.caption(
                            f"Database: {response_json.get('database_name')} | "
                            f"Iterations: {response_json.get('iterations')}"
                        )

                    # ─────────────────────────────
                    # DOCUMENT RESPONSE
                    # ─────────────────────────────
                    elif route == "document":
                        st.subheader("📚 Retrieved Evidence")

                        for i, chunk in enumerate(response_json.get("relevant_chunks", []), 1):
                            st.markdown(f"**Chunk {i}**")
                            st.markdown(chunk.get("content", ""))
                            st.caption(
                                f"Page: {chunk.get('page')} | "
                                f"Section: {chunk.get('section')} | "
                                f"Confidence: {chunk.get('confidence_score')}"
                            )
                            st.divider()

                        st.markdown(
                            f"📜 **Policy Citations:** "
                            f"{response_json.get('policy_citations')}"
                        )

                    # ─────────────────────────────
                    # HYBRID RESPONSE
                    # ─────────────────────────────
                    elif route == "hybrid":
                        st.subheader("🧾 SQL Query Executed")
                        st.code(response_json.get("sql_query_executed", ""), language="sql")

                        st.subheader("📊 Banking Data")
                        banking_data = response_json.get("banking_data")

                        if isinstance(banking_data, list):
                            st.json(banking_data)
                        else:
                            st.write(banking_data)

                        st.subheader("📚 Supporting Policy Evidence")

                        for i, chunk in enumerate(response_json.get("relevant_chunks", []), 1):
                            st.markdown(f"**Chunk {i}**")
                            st.markdown(chunk.get("content", ""))
                            st.caption(
                                f"Page: {chunk.get('page')} | "
                                f"Section: {chunk.get('section')} | "
                                f"Confidence: {chunk.get('confidence_score')}"
                            )
                            st.divider()

                        st.markdown(
                            f"📜 **Policy Citations:** "
                            f"{response_json.get('policy_citations')}"
                        )

                    

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