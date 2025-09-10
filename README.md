# AutoReel 🎬  
**AI-Powered Blog-to-Reel Generator**  

AutoReel is an AI-driven system that automatically converts blog text or short articles into engaging short-form vertical videos (Reels, Shorts, TikToks). The pipeline generates TTS narration, fetches semantically relevant video clips, overlays captions, and stitches them into a creator-ready reel.  

---

## ✨ Features  
- **Text → Video**: Converts any blog or input text into short-form video content.  
- **AI Narration**: Generates natural voiceovers using Text-to-Speech.  
- **Smart Visuals**: Fetches Pexels video clips and applies **CLIP-based semantic reranking** for accurate matches.  
- **Captions Overlay**: Adds readable captions with translucent backgrounds for social media style.  
- **Ken Burns Effect**: Gentle zoom/pan for static clips to enhance engagement.  
- **Streamlit Web App**: Paste text, generate reel, preview output, and download MP4 in one click.  
- **GPU Acceleration**: Supports CUDA for faster CLIP embeddings and reel generation.  

---

## 🛠️ Tech Stack  
- **Programming:** Python  
- **Frameworks & Tools:** Streamlit, dotenv, tqdm  
- **AI/ML:** CLIP (SentenceTransformers), gTTS (Text-to-Speech)  
- **Video Processing:** MoviePy, Pillow  
- **APIs:** Pexels API (for fetching stock media)  
- **Hardware:** NVIDIA GPU acceleration supported  


<img width="1896" height="901" alt="Screenshot 2025-09-10 041832" src="https://github.com/user-attachments/assets/196b9063-abe2-4c5b-a590-e30c1f983fa5" />
<img width="1050" height="702" alt="Screenshot 2025-09-10 051409" src="htt<img width="1082" height="858" alt="Screenshot 2025-09-10 051419" src="https://github.com/user-attachments/assets/4f213c9f-1345-4f42-9c25-e1054f5a61d8" />
ps://github.com/user-attachments/assets/86598074-9183-4d00-b148-811635ec34cb" />
## 🚀 Getting Started  

### 1. Clone the repo  
```bash
git clone https://github.com/your-username/AutoReel.git
cd AutoReel
2. Create environment
conda create -n auto_reel python=3.10 -y
conda activate auto_reel

3. Install dependencies
pip install -r requirements.txt

4. Set up environment variables

Create a .env file in the project root:

PEXELS_API_KEY=your_pexels_api_key_here
NUM_CARDS=5

5. Run the generator (CLI)
python step3_video_clips.py

6. Run the web app (UI)
streamlit run app.py

📂 Project Structure
AutoReel/
├── app.py                 # Streamlit web app
├── step3_video_clips.py   # Core generator pipeline with CLIP reranking
├── sample_input.txt       # Example input text
├── assets/
│   └── bg.jpg             # Fallback background image
├── output/
│   └── reel_clips.mp4     # Example generated reel
├── requirements.txt       # pip dependencies
├── environment.yml        # conda environment (optional)
└── .env.example           # template for API keys

📊 Achievements

Reduced mismatched clip selection by ~60% with CLIP-based reranking.

Generates a 60–90 second reel in under 2 minutes on GPU.

Provides an end-to-end demo-ready web app for creators and hackathon presentations.

🎯 Use Cases

Content creators repurposing blogs into reels.

Hackathon demos showcasing GenAI in creative workflows.

Personal projects or portfolio showcases in AI/ML + media generation.

📸 Demo (Example Prompt)

Input Text:

Success doesn’t happen overnight.  
Tip 1: Build small habits every day.  
Tip 2: Stay consistent, even when it feels hard.  
Tip 3: Learn from mistakes instead of fearing them.  
Remember: tiny steps lead to big wins.


Output: A short-form video reel with narration, captions, and matching visuals.

🤝 Contributing

Pull requests are welcome! If you’d like to add new features (e.g., support for more video sources, better TTS voices), feel free to open an issue first to discuss your idea.
