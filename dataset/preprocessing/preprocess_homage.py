import os
from pathlib import Path

from moviepy.editor import VideoFileClip
from tqdm import tqdm

SOURCE_DIR = Path("/vision/downloads/home_action_genome/hacgen/action_split_data/V1.0")

for root, dirs, files in tqdm(os.walk(SOURCE_DIR)):
    for file in tqdm(files):
        print(file)
        new_root = root.replace("V1.0", "V1.0resized", 1)
        os.makedirs(new_root, exist_ok=True)
        new_file = Path(file).with_suffix(".webm")
        target_path = Path(new_root) / new_file
        if target_path.is_file():
            continue

        full_video = VideoFileClip(os.path.join(root, file)).resize(height=256)
        full_video.write_videofile(str(target_path))

print("DONE")
