import json
import os
from multiprocessing import Pool
from pathlib import Path

import decord
from decord import DECORDError
from moviepy.editor import VideoFileClip
from tqdm import tqdm

SOURCE_DIR = Path("/vision/group/InteractADL_2/ego_view")
ANNOTATIONS_DIR = Path("/vision2/u/jphwa/interactadl/annotations_all_task_240308/atomic_action")
REPLACE_OLD = "vision/group/InteractADL_2/ego_view"
REPLACE_NEW = "vision2/u/jphwa/interactadl/ego_view_actions_resized"
FPS = 25


def process_file(task):
    root, file = task
    if not file.endswith(".mp4"):
        print("skipping", root, file)
        return

    print(root, file)
    task_id = Path(root).parts[-1]
    if task_id in ["task41"]:
        return
    annotation_file = ANNOTATIONS_DIR / f"{task_id}_person{file[-5]}_atomic.json"
    with open(annotation_file) as fp:
        annotations = json.load(fp)

    new_root = Path(root.replace(REPLACE_OLD, REPLACE_NEW, 1)).parent
    full_video = VideoFileClip(os.path.join(root, file)).resize(height=256)
    for i, a in enumerate(annotations):
        target_dir = new_root / a['class'].replace("/", "_or_")
        os.makedirs(target_dir, exist_ok=True)
        target_path = target_dir / f"{task_id}_person{file[-5]}_action{i:03d}.webm"

        # if file can be opened by decord, don't generate it again
        try:
            decord.VideoReader(str(target_path))
            continue
        except (DECORDError, RuntimeError):
            pass

        offset = 0
        while True:
            try:
                start_t = a["frame_start"] / FPS
                end_t = a["frame_end"] / FPS
                if start_t == end_t:
                    end_t = start_t + a['action_length']
                subclip = full_video.subclip(start_t, end_t - offset)
                subclip.write_videofile(str(target_path))
            # except OSError:
            #     offset += 0.01
            #     print(f"setting offset={offset} for {target_path}", flush=True)
            #     continue
            except Exception:
                print(f"failure for {target_path}", flush=True)
                raise
            break


if __name__ == "__main__":
    results = []
    with Pool(processes=8) as pool:
        videos = [(root, file) for root, _, files in os.walk(SOURCE_DIR) for file in files]
        for _ in tqdm(pool.imap_unordered(process_file, videos), desc="OVERALL", total=len(videos)):
            pass

    print("DONE")
