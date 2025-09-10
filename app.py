import os
import streamlit as st
from pathlib import Path
import subprocess

st.set_page_config(page_title="AutoReel Generator", layout="centered")

st.title("🎬 AutoReel — Blog to Reels with AI")

# Input box
user_text = st.text_area("Paste your blog text here:", height=200)

num_cards = st.slider("Number of cards", 3, 8, 5)

generate = st.button("Generate Reel")

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

if generate:
    with open("sample_input.txt", "w", encoding="utf-8") as f:
        f.write(user_text.strip())

    st.info("⏳ Generating reel… this may take 1–2 minutes.")
    result = subprocess.run(
        ["python", "step3_video_clips.py"], capture_output=True, text=True
    )

    if result.returncode != 0:
        st.error("❌ Error generating reel.")
        st.text(result.stderr)
    else:
        out_path = OUTPUT_DIR / "reel_clips.mp4"
        if out_path.exists():
            st.success("✅ Reel generated!")
            st.video(str(out_path))
            with open(out_path, "rb") as f:
                st.download_button(
                    "Download MP4",
                    data=f,
                    file_name="reel.mp4",
                    mime="video/mp4"
                )
