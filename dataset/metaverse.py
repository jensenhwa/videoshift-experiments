import json
from pathlib import Path
from typing import Set

__all__ = ['MetaverseAtWorkActivities', 'MetaverseAtWorkAtomicActions']

from dataset.base import CustomVideoDataset, update_key

METAVERSE_SPLITS_DIR = Path("/vision/u/jphwa/sail_panasonic/videocompare/dataset/splits")


class MetaverseAtWorkActivities(CustomVideoDataset):
    name = 'Metaverse@Work Activities'

    def __init__(self, splits: Set[str], views=('ego', '3rd'), text_type=None):
        self.data_dict = {}
        if not splits <= {"train", "val", "test"}:
            raise NotImplementedError
        if splits != {"test"}:
            raise NotImplementedError("Metaverse dataset should not be used during training;"
                                      "we use this dataset to evaluate generalization ability.")
        splits = {"train", "val", "test"}

        for s in splits:
            for v in views:
                with open(METAVERSE_SPLITS_DIR / f"metaverse_activities_{v}_{s}.json") as fp:
                    data = json.load(fp)
                for key in list(data.keys()):
                    if key in self.data_dict:
                        self.data_dict[key].extend(data[key])
                    else:
                        self.data_dict[key] = data[key]

        self.data_dict = {update_key(k, text_type): v for k, v in self.data_dict.items()}


class MetaverseAtWorkAtomicActions(CustomVideoDataset):
    name = 'Metaverse@Work Atomic Actions'

    def __init__(self, splits: Set[str], views=('ego', '3rd'), text_type=None):
        self.data_dict = {}
        if not splits <= {"train", "val", "test"}:
            raise NotImplementedError
        if splits != {"test"}:
            raise NotImplementedError("Metaverse dataset should not be used during training;"
                                      "we use this dataset to evaluate generalization ability.")
        splits = {"train", "val", "test"}
        for s in splits:
            for v in views:
                with open(METAVERSE_SPLITS_DIR / f"metaverse_{v}_{s}.json") as fp:
                    data = json.load(fp)
                for key in list(data.keys()):
                    if key in self.data_dict:
                        self.data_dict[key].extend(data[key])
                    else:
                        self.data_dict[key] = data[key]

        self.data_dict = {update_key(k, text_type): v for k, v in self.data_dict.items()}
