# asl-video-pipeline
 A full Python pipeline that transcribes speech from video, converts it into ASL gloss, and renders a signing avatar back onto the original video using pose‑based animations.

 # ASL Video Pipeline

A full Python pipeline that:

1. Extracts & transcribes speech from a video (using WhisperX)  
2. Converts English text into ASL gloss (via the OpenAI Chat API)  
3. Generates pose JSONs and renders a signing avatar back onto the original video  

---

## 📦 Prerequisites

- **Python 3.10.8**  
- `git`, `pip`  
- FFmpeg (for MoviePy audio/video I/O)  
- (Optional) OpenPose if you wish to regenerate raw pose data yourself  

---

## 🔧 Installation

```bash
# 1) Clone this repo and enter it
git clone https://github.com/your-username/asl-video-pipeline.git
cd asl-video-pipeline

# 2) Create & activate a Python 3.10.8 virtual environment
python3.10 -m venv .venv

# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 3) Upgrade pip & install dependencies
pip install --upgrade pip
pip install -r requirements.txt


## 🗑️ Cleanup

```bash
rm -rf output/
