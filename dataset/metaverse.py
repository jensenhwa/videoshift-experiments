import glob
import json
from pathlib import Path
from typing import Set

__all__ = ['MetaverseAtWorkActivities', 'MetaverseAtWorkAtomicActions']

METAVERSE_SPLITS_DIR = Path("/vision/u/jphwa/sail_panasonic/videocompare/dataset")


class MetaverseAtWorkActivities:
    name = 'Metaverse@Work Activities'

    def __init__(self, splits: Set[str], views=('ego', '3rd'), text_type=None):
        self.data_dict = {}
        if not splits <= {"train", "val", "test"}:
            raise NotImplementedError

        for s in splits:
            for v in views:
                with open(METAVERSE_SPLITS_DIR / f"metaverse_activities_{v}_{s}.json") as fp:
                    data = json.load(fp)
                for key in list(data.keys()):
                    if key in self.data_dict:
                        self.data_dict[key].extend(data[key])
                    else:
                        self.data_dict[key] = data[key]


class MetaverseAtWorkAtomicActions:
    name = 'Metaverse@Work Atomic Actions'

    def __init__(self, splits: Set[str], views=('ego', '3rd'), text_type=None):
        self.data_dict = {}
        if not splits <= {"train", "val", "test"}:
            raise NotImplementedError
        for s in splits:
            for v in views:
                with open(METAVERSE_SPLITS_DIR / f"metaverse_{v}_{s}.json") as fp:
                    data = json.load(fp)
                for key in list(data.keys()):
                    if key in self.data_dict:
                        self.data_dict[key].extend(data[key])
                    else:
                        self.data_dict[key] = data[key]
