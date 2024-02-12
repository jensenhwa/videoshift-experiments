import json
from pathlib import Path

from typing import Set

__all__ = ['InteractADLActivities', 'InteractADLAtomicActions']

from dataset.base import CustomVideoDataset

IADL_ATOMIC_ACTIONS_DIR = Path("/next/u/rharries/vlm_benchmark.data/InteractADL_egoview_actions_subclips_resized")
IADL_ACTIVITIES_DIR = Path("/next/u/rharries/vlm_benchmark.data/InteractADL_egoview_activities_subclips")


# TODO: Get IADL 3rd-view splits
class InteractADLAtomicActions(CustomVideoDataset):
    name = 'InteractADL Atomic Actions'

    def __init__(self, splits: Set[str], text_type=None):
        self.data_dict = {}

        if not splits <= {"train", "val", "test"}:
            raise NotImplementedError
        for s in splits:
            with open(IADL_ATOMIC_ACTIONS_DIR / "splits" / f"{s}.json", "r") as fp:
                data = json.load(fp)
            for key in list(data.keys()):
                if key in self.data_dict:
                    self.data_dict[key].extend(data[key])
                else:
                    self.data_dict[key] = data[key]

        # Prepend base data dir to relative video paths
        for category, vids in self.data_dict.items():
            for i in range(len(vids)):
                vids[i] = str(IADL_ATOMIC_ACTIONS_DIR / vids[i])


class InteractADLActivities(CustomVideoDataset):
    name = 'InteractADL Activities'

    def __init__(self, splits: Set[str], text_type=None):
        self.data_dict = {}
        if not splits <= {"train", "val", "test"}:
            raise NotImplementedError
        for s in splits:
            with open(IADL_ACTIVITIES_DIR / "splits" / f"{s}.json", "r") as fp:
                data = json.load(fp)
            for key in list(data.keys()):
                if key in self.data_dict:
                    self.data_dict[key].extend(data[key])
                else:
                    self.data_dict[key] = data[key]

        # Prepend base data dir to relative video paths
        for category, vids in self.data_dict.items():
            for i in range(len(vids)):
                vids[i] = str(IADL_ACTIVITIES_DIR / vids[i])
