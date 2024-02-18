import os
from multiprocessing import Pool
from pathlib import Path

import decord
from decord import DECORDError
from moviepy.editor import VideoFileClip
from tqdm import tqdm

#SOURCE_DIR = Path("/vision/downloads/home_action_genome/hacgen/action_split_data/V1.0")
SOURCE_DIR = Path("/vision/group/metaverse_at_work/ego-view_v2")
REPLACE_OLD = "ego-view_v2"
REPLACE_NEW = "ego-view_v2_resized"


def process_file(task):
    root, file = task
    print(file)
    if file[0] == '_':
        return
    new_root = root.replace(REPLACE_OLD, REPLACE_NEW, 1)
    os.makedirs(new_root, exist_ok=True)
    new_file = Path(file).with_suffix(".webm")
    target_path = Path(new_root) / new_file

    # if file can be opened by decord, don't generate it again
    try:
        decord.VideoReader(str(target_path))
        return
    except (DECORDError, RuntimeError):
        pass

    full_video = VideoFileClip(os.path.join(root, file)).resize(height=256)
    full_video.write_videofile(str(target_path))


if __name__ == "__main__":
    results = []
    with Pool(processes=8) as pool:
        videos = [(root, file) for root, _, files in os.walk(SOURCE_DIR) for file in files]
        for _ in tqdm(pool.imap_unordered(process_file, videos), desc="OVERALL", total=len(videos)):
            pass

    print("DONE")
