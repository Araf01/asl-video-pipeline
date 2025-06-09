# asl-video-pipeline
 A full Python pipeline that transcribes speech from video, converts each word into ASL gloss, and renders a signing avatar back onto the original video using pose‑based animations.
 
![Python Version]([https://img.shields.io/badge/python-3.10%2B-blue.svg](https://img.shields.io/badge/python-3.10%2B-blue))
![License](https://img.shields.io/badge/license-MIT-green.svg)

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
git clone https://github.com/Araf01/asl-video-pipeline
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
```
⚙️ Configuration
1. Place all_processed_poses.pkl (from Microsoft’s ASL‑Citizen dataset) into the data/ folder.
Ensure your gloss maps live at:
data/gloss_to_id_map.json
data/id_to_gloss_map.json

2. Adjust VIDEO_INPUT_PATH in scripts/app.py with your ≤ 8 min input video path


📄 License
This project is licensed under the MIT License.


⚠️ Data Notice 
This project uses pose sequences derived from the ASL‑Citizen dataset (Microsoft Research).
Due to that dataset’s licensing policy, all_processed_poses.pkl in this repository.

To generate the required file:

Obtain the ASL-Citizen data directly from Microsoft Research.
Follow the instructions in the dataset's documentation and use the provided helper scripts (createOpenPose.py, train.py) to generate the final all_processed_poses.pkl file.

Place the resulting all_processed_poses.pkl into the data/ folder.

Once in place, the pipeline will automatically generate the JSONs needed for avatar rendering.


