import glob
import json
import os
from collections import defaultdict, Counter
from pathlib import Path
from typing import Tuple, Dict


def split(items, split1: float, split2: float) -> Tuple[Dict, Dict, Dict]:
    train_items = {}
    val_items = {}
    test_items = {}
    for action in items:
        l = len(items[action])
        val_b = round(l * split1)
        if val_b == l:
            val_b -= 1
        success = False
        for offset in [0, -1, 1, -2, 2, -3, 3, -4, 4]:
            if 0 < val_b + offset < l \
                    and items[action][val_b + offset].split(":")[-2:] != items[action][val_b + offset - 1].split(":")[-2:]:
                val_b = val_b + offset
                success = True
                break
        if not success:
            assert l <= 5
            print("split 1 dirty for action", action)
        train_items[action] = items[action][:val_b]
        test_items[action] = items[action][val_b:]

        l = len(train_items[action])
        train_b = round(l * split2)
        if train_b == l:
            train_b -= 1
        success = False
        for offset in [0, -1, 1, -2, 2, -3, 3, -4, 4]:
            if 0 < train_b + offset < l \
                    and train_items[action][train_b + offset].split(":")[-2:] != train_items[action][train_b + offset - 1].split(":")[-2:]:
                train_b = train_b + offset
                success = True
                break
        if not success:
            assert l <= 5
            print("split 2 dirty for action", action)
        val_items[action] = train_items[action][train_b:]
        train_items[action] = train_items[action][:train_b]
    return train_items, val_items, test_items


def split_homage():
    view_counter = Counter()
    items = defaultdict(list)
    root_path = Path("/vision/downloads/home_action_genome/hacgen")
    for root, dirs, files in os.walk(root_path / "annotation_files" / "atomic_actions"):
        for file in files:
            if file.endswith(".json"):
                assert file.endswith("__aa.json")
                parts = file.split("_")
                activity_views = glob.glob(str(root_path / "action_split_data/V1.0" / parts[
                    0] / f"{parts[0]}_{parts[1]}_v???_{parts[2]}.mkv"))
                # if len(activity_views) == 0:
                #     print("stop")
                view_counter.update([len(activity_views)])
                with open(Path(root) / file) as f:
                    for action in json.load(f):
                        items[action['class']].extend([f"{view}:{action['frame_start']}:{action['frame_end']}"
                                                       for view in activity_views])

    train_items, val_items, test_items = split(items, 0.8, 0.75)

    with open("homage_train.json", "w") as train_fp, open("homage_val.json", "w") as val_fp, open("homage_test.json", "w") as test_fp:
        json.dump(train_items, train_fp, indent=4)
        json.dump(val_items, val_fp, indent=4)
        json.dump(test_items, test_fp, indent=4)
    print("DONE!")


def split_metaverse_atomic_actions():
    items = defaultdict(list)
    root_path = Path("/vision/group/metaverse_at_work")
    for root, dirs, files in os.walk(root_path / "atomic_action_v2"):
        for f in files:
            with open(root_path / "atomic_action_v2" / f) as fp:
                act_json = json.load(fp)
            label = act_json["activity"]
            for action in act_json["actions"]:
                # TODO: experiment with 3rd person in conjunction with or instead of ego view
                path = str(root_path / "3rd-view_v2" / f.replace(".json", ".mp4"))
                items[action['class']].append(f"{path}:{action['frame_start']}:{action['frame_end']}")

                # path_ego = str(root_path / "ego-view_v2" / f.replace(".json", ".mp4"))
                # items[action['class']].append(f"{path_ego}:{action['frame_start']}:{action['frame_end']}")

    train_items, val_items, test_items = split(items, 0.8, 0.75)

    with open("metaverse_3rd_train.json", "w") as train_fp, open("metaverse_3rd_val.json", "w") as val_fp, open(
            "metaverse_3rd_test.json", "w") as test_fp:
        json.dump(train_items, train_fp, indent=4)
        json.dump(val_items, val_fp, indent=4)
        json.dump(test_items, test_fp, indent=4)
    print("DONE!")


def split_metaverse_activities():
    items = defaultdict(list)
    root_path = Path("/vision/group/metaverse_at_work")
    for root, dirs, files in os.walk(root_path / "atomic_action_v2"):
        for f in files:
            with open(root_path / "atomic_action_v2" / f) as fp:
                act_json = json.load(fp)
            label = act_json["activity"]
            # TODO: experiment with 3rd person in conjunction with or instead of ego view
            path = str(root_path / "3rd-view_v2" / f.replace(".json", ".mp4"))
            items[label].append(f"{path}")

            # path_ego = str(root_path / "ego-view_v2" / f.replace(".json", ".mp4"))
            # items[label].append(f"{path_ego}")

    train_items, val_items, test_items = split(items, 0.8, 0.75)

    with open("metaverse_activities_3rd_train.json", "w") as train_fp, \
         open("metaverse_activities_3rd_val.json", "w") as val_fp, \
         open("metaverse_activities_3rd_test.json", "w") as test_fp:
        json.dump(train_items, train_fp, indent=4)
        json.dump(val_items, val_fp, indent=4)
        json.dump(test_items, test_fp, indent=4)
    print("DONE!")


if __name__ == "__main__":
    split_metaverse_atomic_actions()
