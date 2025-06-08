import os
import shutil
import json
import glob

import cv2
import numpy as np
from tqdm import tqdm

from pose_predictor import PosePredictor
from avatar_renderer import AvatarRenderer


class SignLanguageComposer:
    """
    Reads a filtered gloss+timestamp JSON (aligned_gloss_json),
    speeds up each sign by speed_factor, chains them back-to-back,
    overlays their avatar animations onto the original video,
    reports any dropped entries, and then cleans up only that JSON plus temps.
    """

    def __init__(
        self,
        original_video_path: str,
        aligned_gloss_json: str,
        output_composite_video: str,
        sign_video_dir: str,
        word_timestams_json: str,
        fps: int = 25,
        canvas_size_hw: tuple[int, int] = (720, 1280),
        speed_factor: float = 2,
    ):
        """
        :param original_video_path: Path to the input video file.
        :param aligned_gloss_json:  Path to aligned_gloss_filtered.json (word+timestamps).
        :param output_composite_video: Path for the final signed video.
        :param sign_video_dir:      Directory for PosePredictor to read/write JSON‑frame folders.
        :param fps:                 Frames per second for compositing.
        :param canvas_size_hw:      Avatar canvas size (height, width).
        :param speed_factor:        Multiplier >1.0 to make each sign faster
                                    (e.g. 2.0 makes a 1s sign take 0.5s).
        """
        self.original_video_path    = original_video_path
        self.aligned_gloss_json     = aligned_gloss_json
        self.output_composite_video = output_composite_video
        self.word_timestams_json    = word_timestams_json
        self.sign_video_dir         = sign_video_dir
        self.fps                    = fps
        self.canvas_size_hw         = canvas_size_hw
        self.speed_factor           = speed_factor
        self.working_copy_path      = "video_with_signs_temp.mp4"
        self.pose_predictor         = PosePredictor()

    def run(self):
        # 1) Copy original to a working file
        shutil.copy(self.original_video_path, self.working_copy_path)

        # 2) Load filtered gloss + timestamps
        with open(self.aligned_gloss_json, "r", encoding="utf-8") as f:
            entries = json.load(f)["word_timestamps"]
        total_entries = len(entries)

        # 3) Build & speed‑up each pose sequence
        occs = []
        for e in entries:
            word = e.get("word", "").strip()
            if not word:
                continue

            # compute start frame
            sf = int(round(e["start"] * self.fps))
            safe = word.replace(" ", "_").lower()
            json_dir = os.path.join(
                self.pose_predictor.OUTPUT_JSON_BASE_DIR,
                f"{safe}_frames"
            )

            # generate poses if missing
            if not os.path.isdir(json_dir) or not os.listdir(json_dir):
                if not self.pose_predictor.get_and_save_poses_for_gloss(word):
                    continue

            # load full sequence
            renderer = AvatarRenderer(
                input_json_dir=json_dir,
                output_video_dir=self.sign_video_dir,
                output_video_name="unused.mp4",
                fps=self.fps,
                canvas_size_hw=self.canvas_size_hw
            )
            full_seq = renderer._load_pose_sequence(json_dir)
            if full_seq.size == 0:
                continue

            # speed‑up via subsampling
            n = full_seq.shape[0]
            new_n = max(1, int(n / self.speed_factor))
            indices = np.minimum(
                (np.arange(new_n) * self.speed_factor).astype(int),
                n - 1
            )
            seq = full_seq[indices]

            occs.append({
                "sf":      sf,
                "ef":      sf + seq.shape[0],
                "seq":     seq,
                "length":  seq.shape[0],
                "renderer": renderer
            })

        dropped = total_entries - len(occs)
        if dropped:
            print(f"Dropped {dropped} entries (no poses available).")
        print(f"Scheduling {len(occs)} animations from {total_entries} entries.")

        if not occs:
            print("No animations to run.")
            return

        # 4) Chain so they play back-to-back
        occs.sort(key=lambda o: o["sf"])
        prev_end = 0
        for o in occs:
            o["sf"] = max(o["sf"], prev_end)
            o["ef"] = o["sf"] + o["length"]
            prev_end = o["ef"]

        # 5) Composite onto video
        cap   = cv2.VideoCapture(self.working_copy_path)
        W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc= cv2.VideoWriter_fourcc(*"mp4v")
        out   = cv2.VideoWriter(self.output_composite_video, fourcc, self.fps, (W, H))

        ptr = 0
        for idx in tqdm(range(total), desc="Composing Final Video"):
            ret, frame = cap.read()
            if not ret:
                break

            # advance past finished
            while ptr < len(occs) and idx >= occs[ptr]["ef"]:
                ptr += 1

            # overlay if active
            if ptr < len(occs):
                o = occs[ptr]
                if o["sf"] <= idx < o["ef"]:
                    rel = idx - o["sf"]
                    avatar = o["renderer"]._draw_frame(o["seq"][rel])
                    ah, aw = avatar.shape[:2]

                    # bottom‑center, shifted up 5%
                    y0 = H - ah - int(0.05 * ah)
                    x0 = (W - aw) // 2

                    y0 = max(0, min(y0, H - ah))
                    x0 = max(0, min(x0, W - aw))
                    h_avail = min(ah, H - y0)
                    w_avail = min(aw, W - x0)

                    if h_avail > 0 and w_avail > 0:
                        roi  = frame[y0:y0+h_avail, x0:x0+w_avail]
                        clip = avatar[:h_avail, :w_avail]
                        mask = (clip.sum(axis=2) > 0)
                        roi[mask] = clip[mask]
                        frame[y0:y0+h_avail, x0:x0+w_avail] = roi

            out.write(frame)

        cap.release()
        out.release()
        print(f"Final video saved to: {self.output_composite_video}")

        # 6) Cleanup
        for path in (self.aligned_gloss_json, self.working_copy_path, self.word_timestams_json):
            if os.path.exists(path):
                os.remove(path)
        for d in glob.glob(os.path.join(self.pose_predictor.OUTPUT_JSON_BASE_DIR, "*_frames")):
            shutil.rmtree(d, ignore_errors=True)

        print("Cleaned up filtered JSON and temporary files.")
