import itertools
import json
import os
from typing import Optional, Union, Iterable, Dict, List

import numpy as np
import torch
from tqdm.autonotebook import tqdm

from dataset import get_dataset_by_name
from model.SimilarityVLM import SimilarityVLM

'''
Handler which loads the information for all supported datasets, and can
produce various formats of iterable datasets for testing.

TODO: Remove moma repo as submodule, instead add instructions to clone it (anywhere) and install it into each VLM environment.
'''

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
MOMA_REPO = os.path.join(FILE_DIR, "moma")

KINETICS_100_DIR = "/home/datasets/kinetics_100"
SMSM_DIR = "/home/datasets/smsm_cmn"
MOMA_DIR = "/home/datasets/moma"

DEFAULT_MIN_TRAIN_VIDS = 1


REPLACEMENTS = {
    "_": " ",
    "sth": "something",
    "swh": "somewhere"
}


def update_key(k: str, text_type: str or None) -> str:
    for old, new in REPLACEMENTS.items():
        k = k.replace(old, new)

    return k


class CustomVideoDataset:
    data_dict: Dict[str, List[str]]


'''
Simple dataset for filling video embedding caches.
This just iterates through all videos referenced in the given dataset split.
Each element is a single video, referenced as a file path.
'''


class SequentialVideoDataset(torch.utils.data.Dataset):
    '''
    Args:
        data_dict ({str -> [str]}): Dictionary from class names to lists of video paths in that class.
    '''

    def __init__(self, data_dict: dict) -> None:
        super().__init__()

        self.video_paths = list(itertools.chain(*data_dict.values()))

    def __getitem__(self, i):
        return self.video_paths[i]

    def __len__(self):
        return len(self.video_paths)


'''
Simple dataset for filling text embedding caches.
This just iterates through all videos referenced in the given dataset split.
'''


class SequentialCategoryNameDataset(torch.utils.data.Dataset):
    '''
    Args:
        data_dict ({str -> [str]}): Dictionary from class names to lists of video paths in that class.
    '''

    def __init__(self, data_dict: dict) -> None:
        super().__init__()

        self.category_names = list(data_dict.keys())

    def __getitem__(self, i):
        return self.category_names[i]

    def __len__(self):
        return len(self.category_names)


class DatasetHandler:
    def __init__(self, name: Union[str, Iterable], split: str = "val", split_type: str = "video", class_limit: Optional[int] = None,
            min_train_videos: int = DEFAULT_MIN_TRAIN_VIDS, label_verb: bool = False):
        self.name = name
        self.split = split
        self.split_type = split_type
        self.class_limit = class_limit
        self.min_train_videos = min_train_videos

        if split not in ["train", "val", "test", "all"]:
            raise ValueError(f"Invalid dataset split: {split}")

        if split_type not in ["class", "video"]:
            raise ValueError(f"Invalid split type: {split_type}")

        if class_limit is not None and class_limit <= 0:
            raise ValueError(f"Class limit must be positive or None. Got {class_limit}.")

        '''
        Populate self.data_dict.
            Keys are category names
            Values are lists of all video paths associated with that category name.
        '''
        if isinstance(name, str):
            self.data_dict = get_dataset_by_name(name, splits={split}).data_dict
        else:
            self.data_dict = {}
            for dataset in name:
                data = get_dataset_by_name(dataset, splits={split}).data_dict
                for key in list(data.keys()):
                    if key in self.data_dict:
                        self.data_dict[key].extend(data[key])
                    else:
                        self.data_dict[key] = data[key]

        if label_verb:
            verb_dict = {}
            for key, value in self.data_dict.items():
                if key.split()[0] in verb_dict:
                    verb_dict[key.split()[0]].extend(value)
                else:
                    verb_dict[key.split()[0]] = value
            self.data_dict = verb_dict

        # Artificially limit the number of classes after the fact
        if self.class_limit is not None and self.class_limit < len(self.data_dict):
            for extra_class in list(self.data_dict.keys())[self.class_limit:]:
                del self.data_dict[extra_class]

        # Remove classes which have too few training examples
        # min_train_videos field only has an effect on split_type="video" datasets, where the classes are the same across splits
        if self.split_type == "video" and min_train_videos > 1:
            # TODO: Determine better way to make this consistent across splits
            if split == "train":
                for cat in list(self.data_dict.keys()):
                    if len(self.data_dict[cat]) < min_train_videos:
                        del self.data_dict[cat]
            elif split == "val":
                train_dataset = DatasetHandler(name, split="train", split_type=split_type, class_limit=class_limit,
                                               min_train_videos=min_train_videos)
                for cat in list(self.data_dict.keys()):
                    if cat not in train_dataset.data_dict.keys():
                        del self.data_dict[cat]
    def id(self) -> str:
        if self.split_type == "class":
            out = f"{self.name}.c.{self.split}"
        else:
            out = f"{self.name}.v.{self.split}"

        # Only include extra info if these uncommon vars are non-default
        if self.min_train_videos != DEFAULT_MIN_TRAIN_VIDS or self.class_limit is not None:
            out += f".vidmin_{self.min_train_videos}"
            out += f".classmax_{self.class_limit}"

        return out

    def category_count(self) -> int:
        return len(self.data_dict)

    def video_count(self) -> int:
        return sum(len(vids) for vids in self.data_dict.values())

    def sequential_video(self) -> SequentialVideoDataset:
        return SequentialVideoDataset(self.data_dict)

    def sequential_category_name(self) -> SequentialCategoryNameDataset:
        return SequentialCategoryNameDataset(self.data_dict)

    def fill_cache(self, vlm: SimilarityVLM) -> None:
        """Triggers the given vlm to generate embeddings for every video and text referenced
        in this dataset split, saving the resulting cache both disk and mem.

        Args:
            vlm (SimilarityVLM): VLM to fill the cache for
        """

        video_dataset = self.sequential_video()
        for i, vid_path in enumerate(tqdm(video_dataset, leave=False)):
            vlm.get_video_embeds(vid_path)

        text_dataset = self.sequential_category_name()
        for i, text in enumerate(tqdm(text_dataset, leave=False)):
            vlm.get_text_embeds(text)

    def export_embeddings(self, vlm: SimilarityVLM, save_dir_path: str) -> None:
        """Computes embeddings for each text and video in the dataset, saving them
        as numpy arrays in .npy format.
        Output Files (in <save_dir_path>):
            category_names.npy:             1D array of text names for each category. Category index
                                            is consistent across all files.
            category_name_embeddings.npy:   2D array of text embeddings for each category. Dim 1 is category index,
                                            dim 2 is embedding dim.
            video_category_indices.npy:     1D array of category indices (corresponding to category_names.npy) for
                                            each video path/embedding.
            video_paths.npy:                1D array of video paths for each video.
            video_embeddings.npy:           2D array of video embeddings for each video.
                                            Dim 1 is video index, dim 32 is embedding dim.
            vlm_info.json:                  Class and parameters for the VLM instance used.

        Args:
            vlm (SimilarityVLM): _description_
            save_dir_path (str): _description_
        """
        self.fill_cache(vlm)

        os.makedirs(save_dir_path, exist_ok=True)

        category_names = np.array(list(self.data_dict.keys()))
        np.save(os.path.join(save_dir_path, "category_names.npy"), category_names)

        category_name_embeddings = np.array([
            vlm.get_text_embeds(name)
            for name in category_names
        ])
        np.save(os.path.join(save_dir_path, "category_name_embeddings.npy"), category_name_embeddings)

        video_category_indices = []
        video_paths = []
        video_embeddings = []
        for i, name in enumerate(category_names):
            video_category_indices += [i] * len(self.data_dict[name])
            video_paths += self.data_dict[name]
            video_embeddings += [
                vlm.get_video_embeds(path)
                for path in self.data_dict[name]
            ]
        video_category_indices = np.array(video_category_indices)
        video_paths = np.array(video_paths)
        video_embeddings = np.array(video_embeddings)

        np.save(os.path.join(save_dir_path, "video_category_indices.npy"), video_category_indices)
        np.save(os.path.join(save_dir_path, "video_paths.npy"), video_paths)
        np.save(os.path.join(save_dir_path, "video_embeddings.npy"), video_embeddings)

        vlm_info_dict = vlm.params()
        vlm_info_dict["class"] = vlm.__class__.__name__
        with open(os.path.join(save_dir_path, "vlm_info.json"), "w") as fp:
            json.dump(vlm_info_dict, fp, indent=2)
