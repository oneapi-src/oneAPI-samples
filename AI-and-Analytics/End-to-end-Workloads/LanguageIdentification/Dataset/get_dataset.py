import os
import shutil
import argparse
from datasets import load_dataset
from tqdm import tqdm

language_to_code = {
    "japanese": "ja",
    "swedish": "sv-SE"
}

def download_dataset(output_dir):
    for lang, lang_code in language_to_code.items():
        print(f"Processing dataset for language: {lang_code}")

        # Load the dataset for the specific language
        dataset = load_dataset("mozilla-foundation/common_voice_11_0", lang_code, split="train")

        # Create a language-specific output folder
        output_folder = os.path.join(output_dir, lang, lang_code, "clips")
        os.makedirs(output_folder, exist_ok=True)

        # Extract and copy MP3 files
        for sample in tqdm(dataset, desc=f"Extracting and copying MP3 files for {lang}"):
            audio_path = sample['audio']['path']
            shutil.copy(audio_path, output_folder)

    print("Extraction and copy complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and copy audio files from a dataset to a specified directory.")
    parser.add_argument("--output_dir", type=str, default="/data/commonVoice", help="Base output directory for saving the files. Default is /data/commonVoice")
    args = parser.parse_args()

    download_dataset(args.output_dir)