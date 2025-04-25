import streamlit as st
import requests

API_BASE = "http://localhost:8000"  # Update if running FastAPI elsewhere

st.title("🎬 YouTube Video Summarizer & QA")

# Input for YouTube URL
youtube_url = st.text_input("Enter a YouTube video URL")

if youtube_url and st.button("Process Video"):
    with st.spinner("Processing video and generating embeddings..."):
        res = requests.post(f"{API_BASE}/process_video", json={"youtube_url": youtube_url})
        if res.status_code == 200:
            st.success("Video processed successfully!")
            st.session_state.video_id = res.json()["video_id"]
        else:
            st.error(res.json().get("detail", "Something went wrong."))

# Ask questions if video is processed
if "video_id" in st.session_state:
    st.markdown("---")
    st.subheader("Ask Questions about the Video")
    user_question = st.text_input("Type your question")

    if user_question:
        with st.spinner("Thinking..."):
            res = requests.post(f"{API_BASE}/ask", json={
                "video_id": st.session_state.video_id,
                "question": user_question
            })
            if res.status_code == 200:
                st.markdown(f"**Answer:** {res.json()['answer']}")
            else:
                # st.error(res.json().get("detail", "Error fetching answer."))
                try:
                   error_message = res.json().get("detail", "Error fetching answer.")
                except Exception:
                    error_message = res.text or "Unknown error (non-JSON response)"
                    st.error(error_message)

