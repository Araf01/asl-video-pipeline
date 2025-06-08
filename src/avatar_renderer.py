import os
import json
import numpy as np
import cv2
from tqdm import tqdm

class AvatarRenderer:
    """
    Loads per-frame OpenPose JSONs for a given gloss and renders them into a video,
    with optional speed‑up of the animation via speed_factor.
    """

    #Default Configuration
    DEFAULT_FPS = 25
    DEFAULT_CANVAS_SIZE_HW = (720, 1280)  # (height, width)
    DEFAULT_SPEED_FACTOR = 1.15           # 1.0 = normal speed, >1.0 = faster

    # Keypoint counts
    KP_COUNT_BODY = 25
    KP_COUNT_HAND = 21
    KP_COUNT_FACE = 70

    TOTAL_EXPECTED_KEYPOINTS = KP_COUNT_BODY + 2 * KP_COUNT_HAND + KP_COUNT_FACE
    NUM_COORDS = 2  # (x, y)

    # Body limbs
    BODY_LIMBS_INDICES = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11),
        (0, 12), (12, 13), (13, 14),
        (1, 15), (0, 15), (0, 16), (1, 16),
        (15, 17), (16, 18),
        (9, 12),
        (11, 24), (11, 22), (22, 23),
        (14, 21), (14, 19), (19, 20),
    ]

    # Hand limbs: build explicitly to avoid scoping issues
    HAND_LIMBS_INDICES = []
    for i in range(5):  # 5 fingers
        base_idx = i * 4
        for j in range(3):
            HAND_LIMBS_INDICES.append((base_idx + j, base_idx + j + 1))


    def __init__(
        self,
        input_json_dir: str,
        output_video_dir: str,
        output_video_name: str,
        fps: int = None,
        canvas_size_hw: tuple[int, int] = None,
        center_y_factor: float = 0.3,
        speed_factor: float = None,
    ):
        """
        :param input_json_dir: Directory with per-frame OpenPose JSONs.
        :param output_video_dir: Directory where rendered video will be saved.
        :param output_video_name: Filename for the rendered video.
        :param fps: Frames per second (defaults to DEFAULT_FPS).
        :param canvas_size_hw: (height, width) of rendering canvas.
        :param center_y_factor: vertical bias for skeleton origin (0.0–1.0).
        :param speed_factor: >1.0 to speed up (e.g. 1.15 = 15% faster), defaults to DEFAULT_SPEED_FACTOR.
        """
        self.input_json_dir    = input_json_dir
        self.output_video_dir  = output_video_dir
        self.output_video_name = output_video_name
        self.fps               = fps if fps is not None else self.DEFAULT_FPS
        self.canvas_size_hw    = (
            canvas_size_hw if canvas_size_hw is not None else self.DEFAULT_CANVAS_SIZE_HW
        )
        self.center_y_factor   = center_y_factor
        self.speed_factor      = (
            speed_factor if speed_factor is not None else self.DEFAULT_SPEED_FACTOR
        )

        os.makedirs(self.output_video_dir, exist_ok=True)
        self.output_video_path = os.path.join(self.output_video_dir, self.output_video_name)


    def _draw_frame(self, all_joints_xy: np.ndarray) -> np.ndarray:
        """
        Draw a single frame of the avatar, with origin shifted by center_y_factor.
        """
        img_h, img_w = self.canvas_size_hw
        img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        cx = img_w // 2
        cy = int(img_h * self.center_y_factor)

        valid = ~np.all(all_joints_xy == 0, axis=1)
        pts = all_joints_xy.copy()
        pts[:, 0] += cx
        pts[:, 1] += cy

        # Split keypoints
        offset = 0
        body_kps   = pts[offset:offset + self.KP_COUNT_BODY]
        body_valid = valid[offset:offset + self.KP_COUNT_BODY]
        offset += self.KP_COUNT_BODY

        hand_l_kps   = pts[offset:offset + self.KP_COUNT_HAND]
        hand_l_valid = valid[offset:offset + self.KP_COUNT_HAND]
        offset += self.KP_COUNT_HAND

        hand_r_kps   = pts[offset:offset + self.KP_COUNT_HAND]
        hand_r_valid = valid[offset:offset + self.KP_COUNT_HAND]
        offset += self.KP_COUNT_HAND

        face_kps   = pts[offset:offset + self.KP_COUNT_FACE]
        face_valid = valid[offset:offset + self.KP_COUNT_FACE]

        # Draw body limbs
        for i, j in self.BODY_LIMBS_INDICES:
            if body_valid[i] and body_valid[j]:
                p1 = (int(body_kps[i, 0]), int(body_kps[i, 1]))
                p2 = (int(body_kps[j, 0]), int(body_kps[j, 1]))
                cv2.line(img, p1, p2, (0, 255, 255), 2)
        # Draw body joints
        for i, p in enumerate(body_kps):
            if body_valid[i]:
                cv2.circle(img, (int(p[0]), int(p[1])), 4, (0, 255, 255), -1)

        # Draw hands
        for kps, vmask, color in [
            (hand_l_kps, hand_l_valid, (255, 0, 255)),
            (hand_r_kps, hand_r_valid, (0, 255, 0)),
        ]:
            for i, j in self.HAND_LIMBS_INDICES:
                if vmask[i] and vmask[j]:
                    p1 = (int(kps[i, 0]), int(kps[i, 1]))
                    p2 = (int(kps[j, 0]), int(kps[j, 1]))
                    cv2.line(img, p1, p2, color, 2)
            for i, p in enumerate(kps):
                if vmask[i]:
                    cv2.circle(img, (int(p[0]), int(p[1])), 4, color, -1)

        # Draw face keypoints
        for i, p in enumerate(face_kps):
            if face_valid[i]:
                cv2.circle(img, (int(p[0]), int(p[1])), 2, (255, 255, 255), -1)

        return img


    def _load_pose_sequence(self, json_dir_path: str) -> np.ndarray:
        """
        Reads all "*_keypoints.json" files, extracts (x,y) coords,
        and returns a numpy array of shape (N_frames, TOTAL_EXPECTED_KEYPOINTS, 2).
        """
        if not os.path.isdir(json_dir_path):
            print(f"JSON directory not found: {json_dir_path}")
            return np.array([])

        files = sorted(f for f in os.listdir(json_dir_path) if f.endswith("_keypoints.json"))
        if not files:
            print(f"No keypoint JSONs in {json_dir_path}")
            return np.array([])

        seq = []
        for fn in tqdm(files, desc="Loading JSONs"):
            with open(os.path.join(json_dir_path, fn), "r", encoding="utf-8") as jf:
                data = json.load(jf)
            people = data.get("people", [])
            if not people:
                seq.append(np.zeros((self.TOTAL_EXPECTED_KEYPOINTS, 2), dtype=np.float32))
                continue

            person = people[0]
            def extract(kname, num_kp):
                flat = person.get(kname, [0.0] * (num_kp * 3))
                pts = [flat[i:i+2] for i in range(0, len(flat), 3)]
                arr = np.array(pts, dtype=np.float32)[:num_kp]
                if arr.shape[0] < num_kp:
                    pad = np.zeros((num_kp - arr.shape[0], 2), dtype=np.float32)
                    arr = np.vstack((arr, pad))
                return arr

            body_xy   = extract("pose_keypoints_2d", self.KP_COUNT_BODY)
            hand_l_xy = extract("hand_left_keypoints_2d", self.KP_COUNT_HAND)
            hand_r_xy = extract("hand_right_keypoints_2d", self.KP_COUNT_HAND)
            face_xy   = extract("face_keypoints_2d", self.KP_COUNT_FACE)

            seq.append(np.concatenate([body_xy, hand_l_xy, hand_r_xy, face_xy], axis=0))

        return np.stack(seq, axis=0)


    def render(self):
        """
        Loads the pose sequence, speeds it up by speed_factor,
        draws each frame, and writes the output video.
        """
        # Load frames
        frames = self._load_pose_sequence(self.input_json_dir)
        if frames.size == 0:
            print("No pose frames found. Aborting.")
            return

        # Speed up via subsampling
        n = frames.shape[0]
        new_n = max(1, int(n / self.speed_factor))
        idxs = np.minimum((np.arange(new_n) * self.speed_factor).astype(int), n - 1)
        fast_frames = frames[idxs]

        # Write video
        height, width = self.canvas_size_hw
        writer = cv2.VideoWriter(
            self.output_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (width, height),
        )

        print(f"Rendering {len(fast_frames)} frames @ {self.fps}fps (×{self.speed_factor:.2f} speed)")
        for joints in tqdm(fast_frames, desc="Rendering Frames"):
            img = self._draw_frame(joints)
            writer.write(img)

        writer.release()
        print(f"Avatar video saved: {self.output_video_path}")
