"""
This script automates batch processing of ASL Citizen videos through OpenPose.
For each dataset split (train/val/test), it:
  1. Reads a CSV listing video filenames.
  2. For each video:
     • Verifies the file exists.
     • Creates an output folder named after the video.
     • Skips processing if JSON output already exists.
     • Invokes OpenPose (with body, hand, and face keypoints) via subprocess,
       writing per-frame JSON into the output folder.
  3. Suppresses OpenPose console output for a clean run.
"""

import os
import pandas as pd
import subprocess
from tqdm import tqdm

#CONFIGURATION
# Update these paths for your environment before running:
OPENPOSE_BIN = "openpose.bin_path"           # Path to the OpenPose executable
VIDEO_DIR     = "ASL_Citizen_video_path"      # Folder containing the ASL Citizen videos
SPLITS_DIR    = "ASL_Citizen_csv_path"        # Folder containing train/val/test CSV files
OUTPUT_ROOT   = "Openpose_output_path"        # Root folder where JSON outputs will be saved
MODEL_DIR     = "openpose/models"             # Folder where OpenPose model subfolders (pose/hand/face) reside
SPLITS        = ["train", "val", "test"]      # Dataset splits to process

# ========= PROCESS EACH DATA SPLIT =========
for split in SPLITS:
    print(f"Processing {split.upper()} split...")

    # Path to the CSV listing all videos in this split
    split_path = os.path.join(SPLITS_DIR, f"{split}.csv")
    df = pd.read_csv(split_path)

    # Iterate through each row (each video file entry)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_file = row["Video file"]            # Filename column in CSV
        video_path = os.path.join(VIDEO_DIR, video_file)

        # Skip missing videos
        if not os.path.exists(video_path):
            print(f"Missing video: {video_path}")
            continue

        # Create an output folder named after the video (without extension)
        base_name       = os.path.splitext(video_file)[0]
        output_json_dir = os.path.join(OUTPUT_ROOT, split, base_name)
        os.makedirs(output_json_dir, exist_ok=True)

        # If JSONs already exist, skip to avoid re-processing
        if os.listdir(output_json_dir):
            continue

        # Build the OpenPose command-line arguments:
        cmd = [
            OPENPOSE_BIN,
            "--video",         video_path,
            "--write_json",    output_json_dir,
            "--model_folder",  MODEL_DIR,
            "--display",       "0",   # Disable GUI display
            "--render_pose",   "0",   # Do not overlay skeleton on video
            "--hand",                # Enable hand keypoints
            "--face",                # Enable face keypoints
            "--num_gpu",       "1",   # Use one GPU
            "--num_gpu_start", "0"    # Start with GPU device 0
        ]

        # Run OpenPose, discarding its console output
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("All splits processed and JSON saved")
