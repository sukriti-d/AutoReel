# step3_video_clips.py
"""
AutoReel — video-clip visual upgrade with CLIP reranking
- Uses Pexels API to fetch video clips
- Reranks candidate thumbnails with CLIP (GPU if available)
- Adds captions, Ken-Burns, TTS audio
- Outputs: output/reel_clips.mp4
"""

import os, re, time, textwrap, requests, numpy as np
from io import BytesIO
from dotenv import load_dotenv
from gtts import gTTS
from moviepy.editor import VideoFileClip, ImageClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip
from PIL import Image, ImageDraw, ImageFont
import torch
from sentence_transformers import SentenceTransformer, util

# --- Pillow compatibility shim ---
if not hasattr(Image, "ANTIALIAS"):
    try: Image.ANTIALIAS = Image.Resampling.LANCZOS
    except Exception: Image.ANTIALIAS = Image.LANCZOS if hasattr(Image, "LANCZOS") else 1

# --- Config ---
load_dotenv()
PEXELS_KEY = os.getenv("PEXELS_API_KEY", "").strip()
NUM_CARDS = int(os.getenv("NUM_CARDS", "5"))
ASSETS_DIR = "assets"; OUTPUT_DIR = "output"
BG_IMAGE = os.path.join(ASSETS_DIR, "bg.jpg")
INPUT_TEXT_FILE = "sample_input.txt"

TARGET_W, TARGET_H = 1080, 1920
FONT_SIZE, MAX_CHARS_LINE = 60, 30
CARD_MIN_SEC, CARD_MAX_SEC, PADDING_SEC = 3.0, 6.0, 0.6

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- CLIP ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device for embeddings:", device)
try:
    clip_model = SentenceTransformer("clip-ViT-B-32", device=device)
    print("CLIP model loaded.")
except Exception as e:
    clip_model = None; print("Warning: CLIP not loaded:", e)

# --- Helpers ---
def read_input_text(path):
    if not os.path.exists(path): raise FileNotFoundError(path)
    return open(path, "r", encoding="utf-8").read().strip()

def sentence_tokenize(text):
    return [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+', text.strip().replace("\n"," ")) if s.strip()]

def chunk_sentences_to_n(sentences, n):
    total = sum(len(s) for s in sentences)
    target = max(30, total // n)
    chunks, cur = [], ""
    for s in sentences:
        if not cur: cur = s
        elif len(cur)+1+len(s) <= target or len(chunks)+1>=n: cur += " "+s
        else: chunks.append(cur); cur = s
    if cur: chunks.append(cur)
    while len(chunks)<n:
        i = max(range(len(chunks)), key=lambda x: len(chunks[x]))
        c = chunks.pop(i); mid=len(c)//2; sp=c.rfind(" ",0,mid)
        if sp==-1: chunks.append(c); break
        a,b=c[:sp].strip(),c[sp+1:].strip(); chunks[i:i]=[a,b]
    while len(chunks)>n: chunks[-2]+= " "+chunks.pop()
    return chunks

def pexels_search_videos(query, per_page=4):
    if not PEXELS_KEY: return []
    try:
        r=requests.get("https://api.pexels.com/videos/search",
            headers={"Authorization":PEXELS_KEY},
            params={"query":query,"per_page":per_page},timeout=10)
        r.raise_for_status(); return r.json().get("videos",[])
    except: return []

def download_url_to_path(url, out):
    try:
        r=requests.get(url,stream=True,timeout=20); r.raise_for_status()
        with open(out,"wb") as f:
            for c in r.iter_content(1024): f.write(c)
        return True
    except: return False

def pick_best_video_for_query(query, dest_prefix, top_k=4):
    if not PEXELS_KEY: return None
    candidates=pexels_search_videos(query,per_page=top_k)
    if not candidates: return None
    if clip_model is None:
        for vid in candidates:
            for f in vid.get("video_files",[]):
                if f.get("file_type","").lower()=="video/mp4":
                    out=f"{dest_prefix}.mp4"
                    if download_url_to_path(f.get("link"),out): return out
        return None
    thumbs, idxs=[],[]
    for i,vid in enumerate(candidates):
        url=vid.get("image") or (vid.get("video_pictures") or [{}])[0].get("picture")
        if not url: continue
        try:
            img=Image.open(BytesIO(requests.get(url,timeout=10).content)).convert("RGB")
            thumbs.append(img); idxs.append(i)
        except: continue
    if not thumbs: return None
    text_emb=clip_model.encode(query,convert_to_tensor=True)
    img_embs=clip_model.encode(thumbs,convert_to_tensor=True)
    best=idxs[int(util.cos_sim(text_emb,img_embs)[0].argmax())]
    files=candidates[best].get("video_files",[])
    for f in sorted(files,key=lambda x:x.get("height",0),reverse=True):
        if f.get("file_type","").lower()=="video/mp4":
            out=f"{dest_prefix}.mp4"
            if download_url_to_path(f.get("link"),out): return out
    return None

def resize_and_crop_image(img,w,h):
    scale=max(w/img.width,h/img.height)
    img=img.resize((int(img.width*scale),int(img.height*scale)),Image.ANTIALIAS)
    x=(img.width-w)//2; y=(img.height-h)//2
    return img.crop((x,y,x+w,y+h))

def make_caption_image(text):
    wrapped=textwrap.wrap(text,width=MAX_CHARS_LINE); tb="\n".join(wrapped)
    img=Image.new("RGBA",(TARGET_W,TARGET_H),(0,0,0,0)); d=ImageDraw.Draw(img)
    try: font=ImageFont.truetype("arial.ttf",FONT_SIZE)
    except: font=ImageFont.load_default()
    bbox=d.multiline_textbbox((0,0),tb,font=font,spacing=8)
    tw,th=bbox[2]-bbox[0],bbox[3]-bbox[1]; x,y=(TARGET_W-tw)//2,TARGET_H-th-120
    d.rectangle([x-30,y-30,x+tw+30,y+th+30],fill=(0,0,0,160))
    d.multiline_text((x,y),tb,font=font,fill=(255,255,255),spacing=8)
    path=os.path.join(OUTPUT_DIR,f"cap_{int(time.time()*1000)}.png"); img.save(path); return path

def generate_tts_audio(text,out): gTTS(text=text,lang="en").save(out)

# --- Main pipeline ---
def build_cards(text,n): return [c for c in chunk_sentences_to_n(sentence_tokenize(text),n) if len(c)>8]

def build_reel(chunks):
    clips=[]
    for i,chunk in enumerate(chunks,1):
        idx=f"{i:02d}"; print(f"[{i}/{len(chunks)}] {chunk[:50]}...")
        audio_path=os.path.join(OUTPUT_DIR,f"card_{idx}.mp3"); generate_tts_audio(chunk,audio_path)
        audio=AudioFileClip(audio_path)
        dur=max(CARD_MIN_SEC,min(CARD_MAX_SEC,audio.duration+PADDING_SEC))
        vpath=pick_best_video_for_query(chunk,os.path.join(OUTPUT_DIR,f"card_{idx}_pexels"))
        if vpath:
            try:
                v=VideoFileClip(vpath)
                sub=v.subclip(0,min(v.duration,dur)).resize(height=TARGET_H)
                if sub.w>TARGET_W: x=(sub.w-TARGET_W)//2; sub=sub.crop(x1=x,x2=x+TARGET_W)
                cap=ImageClip(np.array(Image.open(make_caption_image(chunk)))).set_duration(dur).set_pos(("center","bottom"))
                comp=CompositeVideoClip([sub,cap]).set_duration(dur).set_audio(audio)
                clips.append(comp); continue
            except Exception as e: print("Video error:",e)
        # fallback image
        im=resize_and_crop_image(Image.open(BG_IMAGE).convert("RGB"),TARGET_W,TARGET_H)
        base=ImageClip(np.array(im)).set_duration(dur)
        cap=ImageClip(np.array(Image.open(make_caption_image(chunk)))).set_duration(dur).set_pos(("center","bottom"))
        comp=CompositeVideoClip([base,cap]).set_duration(dur).set_audio(audio)
        clips.append(comp)
    final=concatenate_videoclips(clips,method="compose",padding=-0.2)
    out=os.path.join(OUTPUT_DIR,"reel_clips.mp4")
    final.write_videofile(out,fps=24,codec="libx264",audio_codec="aac")
    print("Saved:",out)

def main():
    print("=== AutoReel — CLIP reranking with audio ===")
    text=read_input_text(INPUT_TEXT_FILE)
    build_reel(build_cards(text,NUM_CARDS))

if __name__=="__main__": main()
