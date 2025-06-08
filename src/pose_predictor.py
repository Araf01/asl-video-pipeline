# pose_predictor.py

import os
import json
import pickle
import torch
import torch.nn as nn
from pathlib import Path

class PosePredictor:
    """
    Encapsulates everything from your original test.py:
    - Constants/paths (FFN_MODEL_LOAD_PATH, GLOSS_TO_ID_MAP_LOAD_PATH, etc.)
    - The WordIDClassifierFFN model definition
    - Loading of gloss_to_id_map, id_to_gloss_map, and processed poses store
    - The get_and_save_poses_for_gloss(...) method
    """

    def __init__(self):
        #Determine base directory of this script
        self.PROJECT_ROOT = Path(__file__).resolve().parent.parent
        self.DATA_DIR     = self.PROJECT_ROOT / "data"
        self.MODELS_DIR   = self.PROJECT_ROOT / "models"
        self.OUTPUT_DIR   = self.PROJECT_ROOT / "output" / "predicted_poses_for_avatar_raw"


        #Configuration constants (must match train_ffn_model.py)
        self.GLOSS_TO_ID_MAP_LOAD_PATH     = self.DATA_DIR / "gloss_to_id_map.json"
        self.ID_TO_GLOSS_MAP_LOAD_PATH     = self.DATA_DIR / "id_to_gloss_map.json"
        self.PROCESSED_POSES_STORE_PATH    = self.DATA_DIR / "all_processed_poses.pkl"
        self.FFN_MODEL_LOAD_PATH           = self.MODELS_DIR / "word_id_classifier_ffn.pth"

        self.KEYPOINT_CONFIG = {
            "pose_keypoints_2d":      {"keypoints": 25},
            "hand_left_keypoints_2d":  {"keypoints": 21},
            "hand_right_keypoints_2d": {"keypoints": 21},
            "face_keypoints_2d":       {"keypoints": 70},
        }
        self.TOTAL_KEYPOINTS     = sum(v["keypoints"] for v in self.KEYPOINT_CONFIG.values())
        self.VALUES_PER_KEYPOINT = 3  # (x, y, confidence)

        # FFN hyperparameters (must match your trained model)
        self.EMBEDDING_DIM_FFN = 128
        self.HIDDEN_DIM_FFN    = 256
        self.NUM_FFN_LAYERS    = 3

        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Output directory for generated per-frame JSONs
        self.OUTPUT_JSON_BASE_DIR = self.OUTPUT_DIR / "predicted_poses_for_avatar_raw"

        # Placeholder attributes to be loaded
        self.gloss_to_id_map       = {}
        self.id_to_gloss_map       = {}
        self.ffn_model             = None
        self.processed_poses_store = {}

        #Attempt to load all assets immediately
        self._load_all_assets()

 
    # FFN Model Definition
    class WordIDClassifierFFN(nn.Module):
        def __init__(self, num_unique_words, embedding_dim, hidden_dim, num_layers=3):
            super().__init__()
            self.embedding = nn.Embedding(num_unique_words, embedding_dim)
            layers_list = [nn.Linear(embedding_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.0)]
            for _ in range(num_layers - 1):
                layers_list += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.0)]
            self.hidden_layers = nn.Sequential(*layers_list)
            self.fc_out = nn.Linear(hidden_dim, num_unique_words)

        def forward(self, word_id_tensor):
            # Ensure shape [batch, 1] if a single integer
            if word_id_tensor.ndim == 1 and word_id_tensor.numel() > 0:
                word_id_tensor = word_id_tensor.unsqueeze(-1)
            elif word_id_tensor.ndim == 0:
                word_id_tensor = word_id_tensor.view(1,1)
            embedded = self.embedding(word_id_tensor)
            if embedded.ndim == 3 and embedded.size(1) == 1:
                embedded = embedded.squeeze(1)
            x = self.hidden_layers(embedded)
            return self.fc_out(x)

   
    # Private asset-loading
    def _load_all_assets(self):
        """
        Loads:
          - gloss_to_id_map from GLOSS_TO_ID_MAP_LOAD_PATH
          - id_to_gloss_map from ID_TO_GLOSS_MAP_LOAD_PATH
          - FFN model from FFN_MODEL_LOAD_PATH
          - processed poses dict from PROCESSED_POSES_STORE_PATH
        Raises FileNotFoundError if any required file is missing.
        """
        # 1) Load gloss_to_id_map
        if not os.path.exists(self.GLOSS_TO_ID_MAP_LOAD_PATH):
            raise FileNotFoundError(f"gloss_to_id_map not found at {self.GLOSS_TO_ID_MAP_LOAD_PATH}")
        with open(self.GLOSS_TO_ID_MAP_LOAD_PATH, 'r') as f:
            self.gloss_to_id_map = json.load(f)
        num_unique = len(self.gloss_to_id_map)
        print(f"[PosePredictor] Loaded gloss_to_id_map with {num_unique} entries.")

        # 2) Load id_to_gloss_map (keys come as strings, convert to int)
        if not os.path.exists(self.ID_TO_GLOSS_MAP_LOAD_PATH):
            raise FileNotFoundError(f"id_to_gloss_map not found at {self.ID_TO_GLOSS_MAP_LOAD_PATH}")
        with open(self.ID_TO_GLOSS_MAP_LOAD_PATH, 'r') as f:
            id_to_gloss_raw = json.load(f)
        # Convert keys to int
        self.id_to_gloss_map = {int(k): v for k, v in id_to_gloss_raw.items()}
        print(f"[PosePredictor] Loaded id_to_gloss_map with {len(self.id_to_gloss_map)} entries.")

        # 3) Initialize and load FFN model weights
        self.ffn_model = PosePredictor.WordIDClassifierFFN(
            num_unique_words=num_unique,
            embedding_dim=self.EMBEDDING_DIM_FFN,
            hidden_dim=self.HIDDEN_DIM_FFN,
            num_layers=self.NUM_FFN_LAYERS
        ).to(self.DEVICE)

        if not os.path.exists(self.FFN_MODEL_LOAD_PATH):
            raise FileNotFoundError(f"FFN model file not found at {self.FFN_MODEL_LOAD_PATH}")
        self.ffn_model.load_state_dict(torch.load(self.FFN_MODEL_LOAD_PATH, map_location=self.DEVICE))
        self.ffn_model.eval()
        print(f"[PosePredictor] FFN model loaded successfully from {self.FFN_MODEL_LOAD_PATH}.")

        # 4) Load processed_poses_store
        if not os.path.exists(self.PROCESSED_POSES_STORE_PATH):
            raise FileNotFoundError(f"Processed pose store not found at {self.PROCESSED_POSES_STORE_PATH}")
        with open(self.PROCESSED_POSES_STORE_PATH, 'rb') as f:
            self.processed_poses_store = pickle.load(f)
        print(f"[PosePredictor] Loaded processed pose store with {len(self.processed_poses_store)} sequences.\n")


    # Public prediction method
    def get_and_save_poses_for_gloss(self, gloss_to_predict: str) -> bool:
        """
        Retrieves a pose sequence (x,y,c) for the given uppercase gloss and saves it as per-frame JSONs.
        - gloss_to_predict must be uppercase (e.g., "HELLO").
        - Outputs JSON files to: OUTPUT_JSON_BASE_DIR/<gloss>_frames/
        Returns True on success, False otherwise.
        """
        print(f"\n[PosePredictor] Processing gloss: '{gloss_to_predict}'")

        if gloss_to_predict not in self.gloss_to_id_map:
            print(f"  Error: Gloss '{gloss_to_predict}' not found in gloss_to_id_map. Cannot predict.")
            return False

        input_word_id = self.gloss_to_id_map[gloss_to_predict]

        # Run FFN forward to check (optional sanity check)
        self.ffn_model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor([input_word_id], dtype=torch.long).to(self.DEVICE)
            logits = self.ffn_model(input_tensor)
            predicted_id = torch.argmax(logits, dim=1).item()

        if predicted_id == input_word_id:
            print(f"  FFN confirms Word ID {predicted_id} for '{gloss_to_predict}'.")
        else:
            correct_gloss = self.id_to_gloss_map.get(input_word_id, "N/A")
            predicted_gloss = self.id_to_gloss_map.get(predicted_id, "N/A")
            print(f"  Warning: FFN predicted ID {predicted_id} ('{predicted_gloss}'), "
                  f"but input ID is {input_word_id} ('{correct_gloss}'). Proceeding with input ID.")

        # Retrieve the pre-processed pose sequence (numpy array, shape = [frames, TOTAL_KEYPOINTS, 3])
        pose_seq_np = self.processed_poses_store.get(input_word_id)
        if pose_seq_np is None:
            print(f"  Error: No pose sequence found for ID {input_word_id} ('{gloss_to_predict}').")
            return False

        if pose_seq_np.shape[-1] != self.VALUES_PER_KEYPOINT:
            print(f"  Error: Pose data for '{gloss_to_predict}' has shape {pose_seq_np.shape}, "
                  f"but each keypoint should have {self.VALUES_PER_KEYPOINT} values (x, y, c).")
            return False

        # Prepare output folder: ex/predicted_poses_for_avatar_raw/<gloss>_frames/
        safe_gloss = gloss_to_predict.replace(" ", "_").lower()
        gloss_frames_dir = os.path.join(self.OUTPUT_JSON_BASE_DIR, f"{safe_gloss}_frames")
        os.makedirs(gloss_frames_dir, exist_ok=True)
        print(f"  Saving JSON frames for '{gloss_to_predict}' into {gloss_frames_dir}.")

        num_frames = pose_seq_np.shape[0]
        for idx in range(num_frames):
            frame_kps = pose_seq_np[idx]  # shape = (TOTAL_KEYPOINTS, 3)
            openpose_data = {
                "version": 1.3,
                "people": [{
                    "person_id": [-1],
                    "pose_keypoints_2d": [],
                    "face_keypoints_2d": [],
                    "hand_left_keypoints_2d": [],
                    "hand_right_keypoints_2d": []
                }]
            }

            offset = 0
            for part_name, cfg in self.KEYPOINT_CONFIG.items():
                num_kps = cfg["keypoints"]
                part_array = frame_kps[offset:offset + num_kps]  # shape (num_kps, 3)
                flat_list = []
                for (x, y, c) in part_array:
                    flat_list += [float(x), float(y), float(c)]
                openpose_data["people"][0][part_name] = flat_list
                offset += num_kps

            json_filename = f"{safe_gloss}_{idx:012d}_keypoints.json"
            json_path = os.path.join(gloss_frames_dir, json_filename)
            try:
                with open(json_path, 'w') as jf:
                    json.dump(openpose_data, jf)
            except Exception as e:
                print(f"Error saving frame {idx} JSON: {e}")

        print(f"Finished saving {num_frames} JSON frames for '{gloss_to_predict}'.")
        return True
