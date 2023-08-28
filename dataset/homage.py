import json
import os
from collections import defaultdict
from enum import Enum
from pathlib import Path

import pytorch_lightning as pl
import torchvision.io
from torch.utils.data import Dataset


class Homage:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
    def get_atomic_actions(self):
        items = defaultdict(list)
        for root, dirs, files in os.walk(self.data_dir / "annotation_files" / "atomic_actions"):
            for file in files:
                if file.endswith(".json"):
                    assert file.endswith("__aa.json")
                    with open(file) as f:
                        for action in json.load(f):
                            items[action['class']].append({
                                "file": file[:-9],
                                "frame_start": action['frame_start'],
                                "frame_end": action['frame_end'],
                                "action_length": action['action_length']
                            })
        return items

    def get_actions(self, data_dir):
        items = defaultdict(list)
        with open(self.data_dir / "list_with_activity_labels" / "train_list.csv") as f:
            for line in f:
                vid, label = line.split(',')
                items[label].append(vid)
        return items

    def __getitem__(self, item):
        v, _, _ = torchvision.io.read_video(self.data_dir /
                                  "action_split_data" /
                                  "V1.0" /
                                  self.items[item]['file'][:5] /
                                  self.items[item]['file'] + ".mkv")
        return v, self.items[item]


class HomageActions(Dataset):
    def __init__(self, data_dir, train=True):
        self.items = []
        self.data_dir = Path(data_dir)
        filepath = data_dir / "list_with_activity_labels" / ("train_list.csv" if train else "val_list.csv")
        with open(filepath) as f:
            for line in f:
                video, label = line.split(",")
                self.items.append({
                    "file": video,
                    "class": label,
                })

    def __getitem__(self, item):

        torchvision.io.read_video(self.data_dir / "action_split_data" / "V1.0" / self.items[item]['file'].replace("_", "/"))
        return self.items[item]

    def __len__(self):
        return len(self.items)


class AnnotationType(Enum):
    ACTIVITY = 1
    ATOMIC_ACTION = 2


class HomeActionDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = "/vision/downloads/home_action_genome/hacgen",
                 annotation_type: AnnotationType = AnnotationType.ACTIVITY):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.annotation_type = annotation_type

    def setup(self, stage: str):
        self.data_dict = {}
        if self.annotation_type == AnnotationType.ACTIVITY:
            self.train_items = []
            self.train_classes = {}
            self.val_items = []
            self.val_classes = {}
            self.data_dir = Path(self.data_dir)
            train_filepath = self.data_dir / "list_with_activity_labels" / "train_list.csv"
            with open(train_filepath) as f:
                for line in f:
                    video, label = line.split(",")
                    self.train_items.append({
                        "file": video,
                        "class": label,
                    })
                    if label in self.train_classes:
                        self.train_classes[label].append(video)
                    else:
                        self.train_classes[label] = [video]

            val_filepath = self.data_dir / "list_with_activity_labels" / "val_list.csv"
            with open(val_filepath) as f:
                for line in f:
                    video, label = line.split(",")
                    self.val_items.append({
                        "file": video,
                        "class": label,
                    })
                    if label in self.val_classes:
                        self.val_classes[label].append(video)
                    else:
                        self.val_classes[label] = [video]

            # todo: Iterate through the atomic actions

    def train_dataloader(self):
        return DataLoader()