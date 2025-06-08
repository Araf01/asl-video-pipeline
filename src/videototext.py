import os
import tempfile
import json
import moviepy.editor as mp
import whisperx
import librosa
import noisereduce as nr
import soundfile as sf


class VideoTranscriberX:
    """
    Transcribe a video into text + per-word timestamps using WhisperX.
    Handles the actual structure returned by whisperx.align (a dict with "segments").
    Falls back to approximate timing if necessary.
    """

    MAX_DURATION_SECONDS = 8 * 60  # 8 minutes

    def __init__(self, whisper_model_size: str = "small", device: str = "cpu"):
        print(f"[Init] Loading WhisperX model '{whisper_model_size}' on device '{device}' (float32)…")
        self.whisper_model_size = whisper_model_size
        self.device = device

        # Force float32 on CPU to avoid float16 errors
        self.asr_model = whisperx.load_model(
            self.whisper_model_size,
            device=self.device,
            compute_type="float32"
        )
        print(f"[Init] WhisperX ASR model loaded.\n")

    def transcribe_video_with_timestamps(self, video_path: str) -> dict:
        """
        Main pipeline:
          1. Extract audio (≤ 8 minutes check)
          2. Optionally noise-reduce
          3. Run ASR (segment-level)
          4. Attempt forced alignment
             → Print raw word_segments (dict with "segments")
          5. Parse out each segment["words"] entry into word_timestamps
             or fall back to approximate timing
          6. Print final word_timestamps to console
          7. Cleanup temp files and return results
        """
        # Step 1: Verify and check duration
        print(f"[Step 1] Verifying video exists: {video_path}")
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        print("[Step 1] File exists.\n")

        print("[Step 2] Checking video duration (must be ≤ 8 minutes)…")
        clip = mp.VideoFileClip(video_path)
        duration = clip.duration
        clip.reader.close()
        if clip.audio:
            clip.audio.reader.close_proc()

        minutes = duration / 60
        print(f"[Step 2] Video duration: {minutes:.2f} minutes.")
        if duration > self.MAX_DURATION_SECONDS:
            raise ValueError(f"Video is too long ({minutes:.2f} minutes); must be ≤ 8 minutes.")
        print("[Step 2] Duration OK.\n")

        # Step 3: Extract audio
        print("[Step 3] Extracting audio from video…")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            audio_path = tf.name
        clip = mp.VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
        clip.reader.close()
        clip.audio.reader.close_proc()
        print(f"[Step 3] Audio saved to: {audio_path}\n")

        # Step 4: Noise reduction (toggle on/off)
        use_noise_reduction = True
        if use_noise_reduction:
            print("[Step 4] Applying noise reduction…")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf2:
                clean_audio_path = tf2.name
            y, sr = librosa.load(audio_path, sr=None)
            reduced = nr.reduce_noise(y=y, sr=sr)
            sf.write(clean_audio_path, reduced, sr)
            print(f"[Step 4] Noise-reduced audio saved to: {clean_audio_path}\n")
        else:
            print("[Step 4] Skipping noise reduction; using raw extracted audio.\n")
            clean_audio_path = audio_path

        # Step 5: ASR transcription
        print("[Step 5] Running ASR transcription (segment-level)…")
        result = self.asr_model.transcribe(clean_audio_path)
        segments = result.get("segments", [])
        language = result.get("language", "en")
        print(f"[Step 5] ASR detected language: {language}")
        print(f"[Step 5] Number of segments: {len(segments)}")
        for i, seg in enumerate(segments):
            text = seg.get("text", "").strip()
            s = seg.get("start", 0.0)
            e = seg.get("end", 0.0)
            print(f"    Segment {i}: “{text}”  ({s:.2f} → {e:.2f})")
        print()

        # Reconstruct full transcript
        full_transcript = ""
        for seg in segments:
            t = seg.get("text", "").strip()
            if t:
                full_transcript += (" " if full_transcript else "") + t

        # Step 6: Attempt forced alignment
        print("[Step 6] Attempting forced alignment for word-level timestamps…")
        try:
            align_model, metadata = whisperx.load_align_model(
                language_code=language, device=self.device
            )
            word_align_output = whisperx.align(
                segments, align_model, metadata, clean_audio_path, device=self.device
            )
            # ** Print raw output of whisperx.align **
            print(f"[Step 6] Raw word_segments (alignment output):\n{word_align_output}\n")
            print(f"[Step 6] Type of word_align_output: {type(word_align_output)}")
        except Exception as e:
            print(f"[Step 6] Alignment threw an exception: {e}")
            word_align_output = None
            print("[Step 6] Proceeding with fallback (no alignment data)\n")

        # Step 7: Parse or fallback to approximate timing
        word_timestamps = []
        if isinstance(word_align_output, dict) and "segments" in word_align_output:
            print("[Step 7] Parsing forced-alignment results from 'segments' key…")
            # Iterate each segment in the alignment output
            for seg in word_align_output["segments"]:
                for w in seg.get("words", []):
                    w_text = w.get("word", "").strip()
                    w_start = float(w.get("start", 0.0))
                    w_end = float(w.get("end", 0.0))
                    if not w_text:
                        continue
                    word_timestamps.append({
                        "word": w_text,
                        "start": w_start,
                        "end": w_end
                    })
            print(f"[Step 7] Parsed {len(word_timestamps)} word timestamps from alignment.\n")

        else:
            # No alignment data or not the expected dict format → fallback
            print("[Step 7] No valid 'segments' in alignment output, falling back to approximate timing…")
            for seg in segments:
                seg_text = seg.get("text", "").strip()
                seg_start = seg.get("start", 0.0)
                seg_end = seg.get("end", 0.0)
                if not seg_text:
                    continue

                # Split segment text into words
                words = [w.strip() for w in seg_text.split() if w.strip()]
                n = len(words)
                if n == 0:
                    continue

                seg_duration = seg_end - seg_start
                per_word = seg_duration / n
                for idx, w in enumerate(words):
                    w_start = seg_start + idx * per_word
                    w_end = w_start + per_word
                    word_timestamps.append({
                        "word": w,
                        "start": w_start,
                        "end": w_end
                    })

            print(f"[Step 7] Created {len(word_timestamps)} approximate word timestamps.\n")

        # Step 8: Print final word timestamps to console
        print("[Debug] Final word_timestamps list:")
        if not word_timestamps:
            print("  → (word_timestamps is EMPTY)\n")
        else:
            for idx, entry in enumerate(word_timestamps):
                print(f"  {idx+1}. Word: “{entry['word']}”  Start: {entry['start']:.2f}s  End: {entry['end']:.2f}s")
            print()

        # Step 9: Clean up temp files
        print("[Step 9] Cleaning up temporary files…")
        for path in (audio_path, clean_audio_path):
            try:
                os.remove(path)
                print(f"    Removed {path}")
            except OSError:
                pass
        print("[Step 9] Cleanup done.\n")

        return {
            "transcription": full_transcript,
            "word_timestamps": word_timestamps
        }

    def save_timestamps_to_json(self, timestamps: list, json_path: str):
        """
        Save the word_timestamps list to a JSON file.
        """
        print(f"[Save] Writing {len(timestamps)} word timestamps to: {json_path}")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"word_timestamps": timestamps}, f, ensure_ascii=False, indent=2)
        print("[Save] JSON written successfully.\n")


