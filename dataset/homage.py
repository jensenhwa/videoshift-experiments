import glob
import json
from pathlib import Path
from typing import Set

__all__ = ['HomeActionGenomeActivities', 'HomeActionGenomeAtomicActions']

from dataset.base import CustomVideoDataset

HOMAGE_ATOMIC_ACTION_SPLITS_DIR = Path("/vision/u/jphwa/sail_panasonic/videocompare/dataset/splits")


class HomeActionGenomeActivities(CustomVideoDataset):
    name = 'Home Action Genome Activities'

    def __init__(self, splits: Set[str], text_type=None):
        self.data_dict = {}
        train_filepath = Path("/vision/downloads/home_action_genome/hacgen")

        if not splits <= {"train", "val"}:
            raise NotImplementedError

        for s in splits:
            with open(train_filepath / "list_with_activity_labels" / f"{s}_list.csv") as f:
                for line in f:
                    video, label = line.split(",")
                    parts = video.split("_")
                    videos = glob.glob(str(train_filepath / "action_split_data/V1.0" / parts[
                        0] / f"{parts[0]}_{parts[1]}_v???_{parts[2]}.mkv"))
                    label = label.replace("_", " ")
                    if label in self.data_dict:
                        self.data_dict[label].extend(videos)
                    else:
                        self.data_dict[label] = videos


class HomeActionGenomeAtomicActions(CustomVideoDataset):
    name = 'Home Action Genome Atomic Actions'

    def __init__(self, splits: Set[str], text_type=None):
        self.data_dict = {}
        if not splits <= {"train", "val", "test"}:
            raise NotImplementedError
        for s in splits:
            with open(HOMAGE_ATOMIC_ACTION_SPLITS_DIR / f"homage_{s}.json") as fp:
                data = json.load(fp)
            for key in list(data.keys()):
                if key in self.data_dict:
                    self.data_dict[key].extend(data[key])
                else:
                    self.data_dict[key] = data[key]
