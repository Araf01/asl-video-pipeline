import os
import re
import json
from dotenv import load_dotenv



class ASLGlossConverter:
    """
    - Loads a gloss‐dataset JSON of available tokens (all uppercase).
    - Filters an aligned‐gloss JSON so that:
       • Tokens not in the dataset are removed entirely (after trying suffix‐stripping).
       • Contractions (e.g. WE'LL) expand into valid tokens (WE, WILL) if possible.
       • When matching tokens, ignores a trailing '1' in dataset keys (e.g., THROW1 matches THROW).
       • Strips common English suffixes: ING, ED, S.
       • Strips out punctuation before matching.
    """

    def __init__(self):
        load_dotenv()

        # A set of valid uppercase gloss tokens (loaded from dataset JSON).
        self.available_glosses: set[str] = set()

        # Map contractions (uppercase) to lists of sub‐tokens (also uppercase).
        self.contraction_map: dict[str, list[str]] = {
            "WE'LL":   ["WE", "WILL"],
            "I'M":     ["I", "AM"],
            "DON'T":   ["DO", "NOT"],
            "CAN'T":   ["CAN", "NOT"],
            "WON'T":   ["WILL", "NOT"],
            "IT'S":    ["IT", "IS"],
            "I'LL":    ["I", "WILL"],
            "YOU'RE":  ["YOU", "ARE"],
            "YOU'LL":  ["YOU", "WILL"],
            "THAT'S":  ["THAT", "IS"],
            "THERE'S": ["THERE", "IS"],
            "WHAT'S":  ["WHAT", "IS"],
            "LET'S":   ["LET", "US"],
            # …add more as needed…
        }
   
    def load_dataset(self, dataset_json_path: str) -> None:
        """
        Loads your gloss‐dataset JSON file into a set of available gloss tokens.

        :param dataset_json_path: Path to JSON whose keys are uppercase gloss tokens.
        """
        if not os.path.isfile(dataset_json_path):
            raise FileNotFoundError(f"Dataset JSON not found: {dataset_json_path}")

        with open(dataset_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.available_glosses = set(data.keys())
        print(f"[Dataset] Loaded {len(self.available_glosses)} gloss tokens from {dataset_json_path}.\n")

    def _exists_in_dataset(self, token: str) -> bool:
        """
        Checks if `token` (uppercase) is in the dataset, allowing a trailing '1' in dataset keys.
        For example: 'THROW' matches 'THROW1' in available_glosses.
        """
        if token in self.available_glosses:
            return True
        if (token + "1") in self.available_glosses:
            return True
        return False
    def _strip_common_suffixes(self, token: str) -> str | None:
        """
        Attempts to strip common English suffixes from the uppercase token to find a base match.
        Returns the base token if found, otherwise None.

        Checks suffixes in order: "ING", "ED", "S".
        """
        if token.endswith("ING"):
            base = token[:-3]
            if self._exists_in_dataset(base):
                return base
        if token.endswith("ED"):
            base = token[:-2]
            if self._exists_in_dataset(base):
                return base
        if token.endswith("S"):
            base = token[:-1]
            if self._exists_in_dataset(base):
                return base
        return None
        

    def filter_missing_gloss(self,
                             input_json_path: str,
                             output_json_path: str) -> None:
        """
        Reads an aligned‐gloss‐timestamps JSON (entries use "word" as the key),
        then:
          1) Strips out punctuation from the raw word.
          2) Converts to uppercase.
          3) Keeps entries if the uppercase word matches (or with trailing "1") in the dataset.
          4) Expands known contractions if each sub‐token exists (or with trailing "1").
          5) Keeps multi‐token phrases if every sub‐token matches (or with trailing "1").
          6) Attempts to strip "ING", "ED", or trailing "S" from a single token and re-check.
          7) Drops any entry that fails all the above tests.

        Writes the filtered list to output_json_path.

        :param input_json_path:  Path to aligned‐gloss JSON:
                                 {
                                   "word_timestamps": [
                                     {"word": "Video,",    "start": 1.739, "end": 1.979},
                                     {"word": "we'll",     "start": 1.999, "end": 2.1},
                                     {"word": "Throwing!", "start": 2.2,   "end": 2.5},
                                     {"word": "asking?",   "start": 2.5,   "end": 2.8},
                                     {"word": "talk",      "start": 3.0,   "end": 3.3}
                                   ]
                                 }
        :param output_json_path: Path to write the filtered JSON:
                                 {
                                   "word_timestamps": [
                                     {"word": "VIDEO",    "start": 1.739, "end": 1.979},
                                     {"word": "WE",       "start": 1.999, "end": 2.1},
                                     {"word": "WILL",     "start": 1.999, "end": 2.1},
                                     {"word": "THROW",    "start": 2.2,   "end": 2.5},
                                     {"word": "ASK",      "start": 2.5,   "end": 2.8},
                                     {"word": "TALK",     "start": 3.0,   "end": 3.3}
                                   ]
                                 }
        """
        if not self.available_glosses:
            raise RuntimeError("No dataset loaded. Call load_dataset(...) first.")

        with open(input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        original_list = data.get("word_timestamps", [])
        final_list: list[dict] = []
        dropped_any = False

        for entry in original_list:
            raw_word = entry.get("word", "").strip()
            start = entry.get("start")
            end = entry.get("end")

            if not raw_word:
                dropped_any = True
                continue

            # 1) Remove punctuation (anything that is not alphanumeric or apostrophe)
            cleaned = re.sub(r"[^\w']", "", raw_word)
            upper_word = cleaned.upper()

            # 2) Exact match (single token exists, ignoring trailing '1')
            base = upper_word
            if base.endswith("1"):
                base = base[:-1]
            if self._exists_in_dataset(base):
                final_list.append({"word": base, "start": start, "end": end})
                continue

            # 3) Contraction?
            if upper_word in self.contraction_map:
                sub_tokens = self.contraction_map[upper_word]
                missing_sub = [t for t in sub_tokens if not self._exists_in_dataset(t)]
                if not missing_sub:
                    for st in sub_tokens:
                        final_list.append({"word": st, "start": start, "end": end})
                    continue
                else:
                    print(f"[Filter] Dropping '{raw_word}' ⇒ missing sub‐tokens {missing_sub}")
                    dropped_any = True
                    continue

            # 4) Multi‐token phrase? (e.g. "GO FUTURE")
            sub_tokens = upper_word.split()
            if len(sub_tokens) > 1:
                missing_sub = [t for t in sub_tokens if not self._exists_in_dataset(t)]
                if not missing_sub:
                    final_list.append({"word": upper_word, "start": start, "end": end})
                    continue
                else:
                    print(f"[Filter] Dropping '{raw_word}' ⇒ missing sub‐tokens {missing_sub}")
                    dropped_any = True
                    continue

            # 5) Try stripping common suffixes (ING, ED, S)
            stripped = self._strip_common_suffixes(upper_word)
            if stripped:
                final_list.append({"word": stripped, "start": start, "end": end})
                continue

            # 6) Otherwise: drop the entry entirely
            print(f"[Filter] Dropping '{raw_word}' ⇒ no match (even after suffix stripping)")
            dropped_any = True
            continue

        if not dropped_any:
            print("[Filter] No tokens were dropped; all entries valid.\n")
        else:
            print(f"[Filter] Completed filtering. Processed {len(original_list)} entries; some dropped.\n")

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump({"word_timestamps": final_list}, f, ensure_ascii=False, indent=2)

        print(f"[Filter] Filtered JSON saved to {output_json_path}.\n")


