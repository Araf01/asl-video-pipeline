import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset as TorchDataset, DataLoader
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import pickle


# !!! UPDATE THESE PATHS IN USE CASE!!!
CSV_FILE_PATH = 'ASL_Citizens_scsv_PATH'
OPENPOSE_BASE_FOLDER = 'poenpose_data_path'

# Output file names
FFN_MODEL_SAVE_PATH = "word_id_classifier_ffn.pth"
GLOSS_TO_ID_MAP_SAVE_PATH = "gloss_to_id_map.json"
ID_TO_GLOSS_MAP_SAVE_PATH = "id_to_gloss_map.json"
ID_TO_DATA_INFO_MAP_SAVE_PATH = "id_to_data_info_map.json"
PROCESSED_POSES_STORE_PATH = "all_processed_poses.pkl"

# Keypoint configuration (TARGETING 137 KEYPOINTS FOR RENDERER COMPATIBILITY)
# For this version, ref_idx_for_norm is not used as we are storing raw values
KEYPOINT_CONFIG = {
    "pose_keypoints_2d": {"keypoints": 25},
    "hand_left_keypoints_2d": {"keypoints": 21},
    "hand_right_keypoints_2d": {"keypoints": 21},
    "face_keypoints_2d": {"keypoints": 70},
}
TOTAL_KEYPOINTS = sum(kp_info["keypoints"] for kp_info in KEYPOINT_CONFIG.values()) # Should be 137
VALUES_PER_KEYPOINT = 3 # Storing x, y, c (original confidence)

# FFN Model Hyperparameters
EMBEDDING_DIM_FFN = 128
HIDDEN_DIM_FFN = 256
NUM_FFN_LAYERS = 3

# Training Parameters for FFN
FFN_EPOCHS = 100
FFN_BATCH_SIZE = 32
FFN_LEARNING_RATE = 5e-4 # Adjusted from your previous log

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. FFN Model Definition (AI Model) ---
class WordIDClassifierFFN(nn.Module):
    def __init__(self, num_unique_words, embedding_dim, hidden_dim, num_layers=NUM_FFN_LAYERS):
        super().__init__()
        self.embedding = nn.Embedding(num_unique_words, embedding_dim)
        layers_list = []
        layers_list.append(nn.Linear(embedding_dim, hidden_dim))
        layers_list.append(nn.ReLU())
        layers_list.append(nn.Dropout(0.0)) # Set to 0.0 for max memorization
        for _ in range(num_layers - 1):
            layers_list.append(nn.Linear(hidden_dim, hidden_dim))
            layers_list.append(nn.ReLU())
            layers_list.append(nn.Dropout(0.0))
        self.hidden_layers = nn.Sequential(*layers_list)
        self.fc_out = nn.Linear(hidden_dim, num_unique_words)

    def forward(self, word_id_tensor):
        if word_id_tensor.ndim == 1 and word_id_tensor.numel() > 0 :
             word_id_tensor = word_id_tensor.unsqueeze(-1)
        elif word_id_tensor.ndim == 0:
             word_id_tensor = word_id_tensor.view(1,1)
        embedded = self.embedding(word_id_tensor)
        if embedded.ndim == 3 and embedded.size(1) == 1:
            embedded = embedded.squeeze(1)
        x = self.hidden_layers(embedded)
        logits = self.fc_out(x)
        return logits

# --- 2. Simple Dataset for FFN Training ---
class WordIDDataset(TorchDataset):
    def __init__(self, word_ids_list):
        self.word_ids = torch.tensor(word_ids_list, dtype=torch.long)
    def __len__(self):
        return len(self.word_ids)
    def __getitem__(self, idx):
        return self.word_ids[idx], self.word_ids[idx]

# --- 3. Pose Loading and Preprocessing Utility Functions (MODIFIED for raw x,y,c) ---
def find_pose_folder_for_video(video_file_name, gloss_str, pose_base_folder):
    try:
        video_id_part = video_file_name.split('-')[0]
        if not os.path.isdir(pose_base_folder): return None
        potential_folders = [d for d in os.listdir(pose_base_folder) if d.startswith(video_id_part)]
        if not potential_folders: return None
        if len(potential_folders) == 1: return os.path.join(pose_base_folder, potential_folders[0])
        gloss_simple = gloss_str.replace(" ", "").lower()
        matched_by_gloss_too = [pf for pf in potential_folders if gloss_simple in pf.replace("-", "").replace("_", "").lower()]
        if len(matched_by_gloss_too) == 1: return os.path.join(pose_base_folder, matched_by_gloss_too[0])
        return None
    except Exception: return None

def load_and_process_single_pose_sequence_raw(folder_path, keypoint_config_dict_local,
                                             total_kps_config_local=TOTAL_KEYPOINTS,
                                             vals_per_kp_local=VALUES_PER_KEYPOINT):
    if not os.path.isdir(folder_path): return None
    frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith('_keypoints.json')])
    if not frame_files: return None

    all_frames_kps_structured = []

    for frame_file in frame_files:
        json_path = os.path.join(folder_path, frame_file)
        try:
            with open(json_path, 'r') as f: data = json.load(f)
            if not data.get('people') or not data['people']: continue
            person_data = data['people'][0]
            
            frame_kps_parts_list = []
            for part_name, config in keypoint_config_dict_local.items():
                kp_data_flat_raw = person_data.get(part_name, []) # e.g., "pose_keypoints_2d"
                num_expected_kps = config["keypoints"]
                
                part_xync_values = []
                for i in range(0, len(kp_data_flat_raw), 3): # Step by 3 for x,y,c
                    if i + 2 < len(kp_data_flat_raw): # Ensure x, y, c exist
                        part_xync_values.extend([
                            kp_data_flat_raw[i],     # x
                            kp_data_flat_raw[i+1],   # y
                            kp_data_flat_raw[i+2]    # c (original confidence)
                        ])
                
                # Reshape to (N_detected_kps_in_part, 3)
                part_xync_arr = np.array(part_xync_values, dtype=np.float32).reshape(-1, vals_per_kp_local)
                
                # Pad or truncate keypoints within this part to ensure consistent number of keypoints per part
                if part_xync_arr.shape[0] < num_expected_kps:
                    padding_shape = (num_expected_kps - part_xync_arr.shape[0], vals_per_kp_local)
                    padding = np.zeros(padding_shape, dtype=np.float32)
                    part_xync_arr = np.vstack((part_xync_arr, padding)) if part_xync_arr.size > 0 else padding
                elif part_xync_arr.shape[0] > num_expected_kps:
                    part_xync_arr = part_xync_arr[:num_expected_kps, :]
                
                frame_kps_parts_list.append(part_xync_arr)
            
            if not frame_kps_parts_list: continue # Skip frame if no parts were processed
            current_frame_structured_kps = np.vstack(frame_kps_parts_list) # Shape (TOTAL_KEYPOINTS, VALUES_PER_KEYPOINT)
            all_frames_kps_structured.append(current_frame_structured_kps)
        except Exception as e:
            # print(f"Error processing frame {json_path}: {e}") # Optional debug
            continue # Skip problematic frames
            
    if not all_frames_kps_structured: return None
    
    # This is now the raw (but structured and padded per part) x,y,c data
    pose_seq_struct_raw_xync = np.array(all_frames_kps_structured, dtype=np.float32) 

    # Ensure the final array has the expected dimensions for safety
    if pose_seq_struct_raw_xync.ndim != 3 or pose_seq_struct_raw_xync.shape[0] == 0 or \
       pose_seq_struct_raw_xync.shape[1] != total_kps_config_local or \
       pose_seq_struct_raw_xync.shape[2] != vals_per_kp_local:
        # print(f"Warning: Final pose sequence from {folder_path} has unexpected shape: {pose_seq_struct_raw_xync.shape}. Expected (frames, {total_kps_config_local}, {vals_per_kp_local})")
        return None # Return None if shape is not as expected after processing all frames

    # NO NORMALIZATION - We want to store raw (structured) x,y,c values
    
    return pose_seq_struct_raw_xync

#4. Main Execution Block
if __name__ == '__main__':
    print(f"FFN Training and RAW Pose Store Creation Script - Using device: {DEVICE}")

    #Phase A: Prepare Data Mappings
    print("\n--- Phase A: Preparing Data Mappings ---")
    try:
        df_main = pd.read_csv(CSV_FILE_PATH)
        if df_main.empty: raise ValueError(f"CSV file is empty: {CSV_FILE_PATH}")
        if 'Gloss' not in df_main.columns or 'Video file' not in df_main.columns:
            raise ValueError("CSV file must contain 'Gloss' and 'Video file' columns.")
    except Exception as e:
        print(f"Error reading or processing CSV file '{CSV_FILE_PATH}': {e}"); exit()

    unique_glosses_list = sorted(list(df_main['Gloss'].astype(str).unique()))
    gloss_to_id_map = {gloss: i for i, gloss in enumerate(unique_glosses_list)}
    id_to_gloss_map = {i: gloss for gloss, i in gloss_to_id_map.items()}
    num_unique_words = len(unique_glosses_list)
    print(f"Found {num_unique_words} unique glosses.")

    id_to_data_info_map = {}
    for gloss_str_map, word_id_map in gloss_to_id_map.items():
        matching_rows = df_main[df_main['Gloss'] == gloss_str_map]
        if not matching_rows.empty:
            id_to_data_info_map[word_id_map] = {
                "video_file": str(matching_rows.iloc[0]['Video file']),
                "gloss_str": gloss_str_map }
    try:
        os.makedirs("ex", exist_ok=True) # Ensure 'ex' directory exists
        with open(GLOSS_TO_ID_MAP_SAVE_PATH, 'w') as f: json.dump(gloss_to_id_map, f, indent=4)
        id_to_gloss_map_str_keys = {str(k): v for k,v in id_to_gloss_map.items()}
        with open(ID_TO_GLOSS_MAP_SAVE_PATH, 'w') as f: json.dump(id_to_gloss_map_str_keys, f, indent=4)
        id_to_data_info_map_str_keys = {str(k): v for k, v in id_to_data_info_map.items()}
        with open(ID_TO_DATA_INFO_MAP_SAVE_PATH, 'w') as f: json.dump(id_to_data_info_map_str_keys, f, indent=4)
        print(f"Mappings saved to directory 'ex/'")
    except Exception as e: print(f"Error saving mapping files: {e}")

    all_word_ids_for_ffn_training = list(range(num_unique_words))
    if not all_word_ids_for_ffn_training: print("Error: No word IDs to train on."); exit()
    ffn_torch_dataset = WordIDDataset(all_word_ids_for_ffn_training)
    ffn_dataloader = DataLoader(ffn_torch_dataset, batch_size=FFN_BATCH_SIZE, shuffle=True)
    # print(f"Prepared FFN training dataloader with {len(all_word_ids_for_ffn_training)} samples.") # Less verbose

    #Phase B: Train FFN (AI Model)
    print("\n--- Phase B: Training Word ID Classifier FFN ---")
    ffn_classifier_model = WordIDClassifierFFN(num_unique_words, EMBEDDING_DIM_FFN, HIDDEN_DIM_FFN, NUM_FFN_LAYERS).to(DEVICE)
    optimizer_ffn = optim.AdamW(ffn_classifier_model.parameters(), lr=FFN_LEARNING_RATE)
    criterion_ffn = nn.CrossEntropyLoss()
    print(f"FFN Model Hyperparameters: EmbeddingDim={EMBEDDING_DIM_FFN}, HiddenDim={HIDDEN_DIM_FFN}, Layers={NUM_FFN_LAYERS}")
    for epoch in range(1, FFN_EPOCHS + 1):
        ffn_classifier_model.train(); epoch_total_loss = 0.0; epoch_correct_preds = 0; epoch_total_samples = 0
        progress_bar = tqdm(ffn_dataloader, desc=f"FFN Epoch {epoch:02d}/{FFN_EPOCHS}", unit="batch", leave=False)
        for input_batch_ids, target_batch_ids in progress_bar:
            input_batch_ids, target_batch_ids = input_batch_ids.to(DEVICE), target_batch_ids.to(DEVICE)
            optimizer_ffn.zero_grad(); logits = ffn_classifier_model(input_batch_ids)
            loss = criterion_ffn(logits, target_batch_ids); loss.backward(); optimizer_ffn.step()
            epoch_total_loss += loss.item(); preds = torch.argmax(logits, dim=1)
            epoch_correct_preds += (preds == target_batch_ids).sum().item(); epoch_total_samples += target_batch_ids.size(0)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        avg_epoch_loss = epoch_total_loss / len(ffn_dataloader) if len(ffn_dataloader) > 0 else 0
        epoch_accuracy = epoch_correct_preds / epoch_total_samples if epoch_total_samples > 0 else 0
        if epoch % 10 == 0 or epoch == FFN_EPOCHS: # Print every 10 epochs and the last epoch
             print(f"FFN Epoch {epoch:02d}/{FFN_EPOCHS} | Avg Loss: {avg_epoch_loss:.6f} | Accuracy: {epoch_accuracy*100:.2f}%")
    try:
        torch.save(ffn_classifier_model.state_dict(), FFN_MODEL_SAVE_PATH)
        print(f"FFN model trained and saved to {FFN_MODEL_SAVE_PATH}")
    except Exception as e: print(f"Error saving FFN model: {e}")

    #Phase D: Create and Save Processed Pose Data Store (with RAW x,y,c values)
    print(f"\n--- Phase D: Creating Processed Pose Data Store (Raw x,y,c values) ---")
    if not (os.path.exists(OPENPOSE_BASE_FOLDER)):
        print(f"  Error: OPENPOSE_BASE_FOLDER '{OPENPOSE_BASE_FOLDER}' not found. Cannot create pose store.")
    elif 'id_to_data_info_map' not in locals() or not id_to_data_info_map:
        print("  Error: id_to_data_info_map not available or empty. Cannot create pose store.")
    else:
        processed_poses_dictionary = {}
        print(f"  Processing {len(id_to_data_info_map)} unique gloss entries to store their poses...")
        progress_bar_store = tqdm(id_to_data_info_map.items(), total=len(id_to_data_info_map), desc="Storing Raw Poses")
        for word_id, data_info in progress_bar_store:
            gloss_str = data_info["gloss_str"]; video_file = data_info["video_file"]
            progress_bar_store.set_postfix(gloss=gloss_str[:20])
            pose_folder_path = find_pose_folder_for_video(video_file, gloss_str, OPENPOSE_BASE_FOLDER)
            if pose_folder_path:
                pose_sequence_np = load_and_process_single_pose_sequence_raw( # Using new function name
                    pose_folder_path, KEYPOINT_CONFIG, 
                    TOTAL_KEYPOINTS, VALUES_PER_KEYPOINT 
                )
                if pose_sequence_np is not None and pose_sequence_np.shape[0] > 0:
                    processed_poses_dictionary[word_id] = pose_sequence_np
            
        if processed_poses_dictionary:
            try:
                with open(PROCESSED_POSES_STORE_PATH, 'wb') as f_store:
                    pickle.dump(processed_poses_dictionary, f_store)
                print(f"  Successfully created and saved RAW processed pose data store to: {PROCESSED_POSES_STORE_PATH}")
                print(f"    Stored {len(processed_poses_dictionary)} pose sequences.")
            except Exception as e: print(f"  Error saving processed pose data store: {e}")
        else: print("  No poses were processed to store. The store will be empty or not created.")

    print("\nTraining and Pose Store Creation script finished.")