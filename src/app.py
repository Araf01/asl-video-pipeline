# run_asl_converter.py

from text_to_gloss import ASLGlossConverter
from videototext import VideoTranscriberX
from sign_language_composer import SignLanguageComposer
import os
from pathlib import Path

# 1. Define the absolute path to the project's root directory
# This will work no matter where you run the script from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 2. Define paths to your data and video folders relative to the root
DATA_DIR = PROJECT_ROOT / "data"
VIDEOS_DIR = PROJECT_ROOT / "videos"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Configuration Variables
#!!!UPDATE VIDEO PATH!!!
VIDEO_INPUT_PATH         = "VIDEO_PATH" 
WORD_TIMESTAMPS_JSON     = DATA_DIR / "word_timestamps.json"
ALIGNED_GLOSS_JSON       = OUTPUT_DIR / "aligned_gloss_filtered.json"
GLOSS_DATASET_JSON       = DATA_DIR /  "gloss_to_id_map.json"
FINAL_SIGNED_VIDEO_PATH  = OUTPUT_DIR / "final_with_signs.mp4"

# This folder is only needed so PosePredictor can write/read JSON frames temporarily.
SIGN_VIDEO_DIR          = OUTPUT_DIR / "predicted_poses_for_avatar_raw"

os.makedirs(SIGN_VIDEO_DIR, exist_ok=True)

# Step 1: Transcribe video → word_timestamps.json
def videototext():
    transcriber = VideoTranscriberX(whisper_model_size="small", device="cpu")
    try:
        result = transcriber.transcribe_video_with_timestamps(VIDEO_INPUT_PATH)
        transcription = result["transcription"]
        word_timestamps = result["word_timestamps"]

        print("\n=== Full Transcription ===")
        print(transcription)

        transcriber.save_timestamps_to_json(word_timestamps, WORD_TIMESTAMPS_JSON)
        print(f"Timestamps saved to {WORD_TIMESTAMPS_JSON}")
    except Exception as e:
        print(f"Error: {e}")

# Step 2: Convert word_timestamps.json → aligned_gloss_filtered.json
def texttogloss():
    converter = ASLGlossConverter()
    # Load uppercase gloss dataset
    converter.load_dataset(GLOSS_DATASET_JSON)

    # Filter the raw timestamps
    converter.filter_missing_gloss(
        input_json_path=WORD_TIMESTAMPS_JSON,
        output_json_path=ALIGNED_GLOSS_JSON
    )


# Step 3: Compose final video by drawing avatar frames in place
def composesignvideo():
    composer = SignLanguageComposer(
        original_video_path    = VIDEO_INPUT_PATH,
        aligned_gloss_json     = ALIGNED_GLOSS_JSON,  
        output_composite_video = FINAL_SIGNED_VIDEO_PATH,
        sign_video_dir         = SIGN_VIDEO_DIR,
        word_timestams_json    = WORD_TIMESTAMPS_JSON,
        fps                    = 25,
        canvas_size_hw         = (720,1280),
    )
    composer.run()
    print(f"Final signed video available at: {FINAL_SIGNED_VIDEO_PATH}")


# Main Entry Point
if __name__ == "__main__":
    videototext()
    texttogloss()
    composesignvideo()