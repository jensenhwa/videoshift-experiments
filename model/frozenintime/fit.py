import numpy as np
import random
import os
from typing import Optional, List

import torch

from .frozenintime.model.model import sim_matrix, FrozenInTime
import json
from .frozenintime.utils.util import state_dict_data_parallel_fix
import transformers

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from model.SimilarityVLM import SimilarityVLM
from similarity_metrics import Similarity

from torchvision.io import read_video
from pytorchvideo.transforms import *
from torchvision.transforms import Compose, Lambda, CenterCrop, RandomHorizontalFlip, Resize, Normalize

import math
import decord
import pdb

from .frozenintime.model.loss import NormSoftmaxLoss
from dataset.base import DatasetHandler
from dataset.few_shot_dataset import FewShotTaskDataset

# Default cache locations
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
CACHE_NAME = "cache"


class FiTVLM(SimilarityVLM):
    """
    Similarity-based VLM that uses Frozen-in-Time for frame and text encoders.
    """

    def __init__(self, path: str = "model/frozenintime/frozenintime/configs/videocompare.json",
                 num_seconds: int = 2, use_cuda: bool = False,
                 reset_cache: bool = False):

        """
        :param path: Path to the config file
        :param num_seconds: Number of seconds to use in the video during inference, converts to 30fps
        :param use_cuda: Whether to use cuda for GPU (if available), if false uses CPU
        :param reset_cache: Whether to reset the embedding cache
        """

        self.path = str(path)  # Pretrained model identifier
        self.config = json.load(open(path))
        self.num_seconds = int(num_seconds)
        self.use_cuda = bool(use_cuda)
        self.loss = NormSoftmaxLoss()

        self.model = None
        #Do not load model, this is just dummy model to access methods
        if path is None:
            print("Dummy model loaded, no backbone or weights!")
            return

        assert type(self.path) is str
        assert type(self.num_seconds) is int

        # Load model
        self.load_model(path=self.path)

        self.cuda = use_cuda and DEVICE == "cuda"
        self.transforms = self.get_transforms()
        self.train_transforms = self.get_train_transforms()
        decord.bridge.set_bridge("torch")  # Video loader

        text_model_name = self.config['arch']['args']['text_params']['model']
        if "openai/clip" in text_model_name:
            tokenizer_builder = transformers.CLIPTokenizer
        else:
            tokenizer_builder = transformers.AutoTokenizer
        self.tokenizer = tokenizer_builder.from_pretrained(
            text_model_name,
            model_max_length=1e6,
            TOKENIZERS_PARALLELISM=False)

        super().__init__(cache_file=os.path.join(FILE_DIR, CACHE_NAME), reset_cache=reset_cache)

    def params(self) -> dict:
        """
        Specify the value of all VLM-specific parameters which may affect prediction accuracy.
        This is used to differentiate test results which use different versions of the same VLM.
        :return:
        :rtype: dict
        """
        return {
            "path": self.path,
            "num_seconds": self.num_seconds,
            "use_cuda": self.use_cuda
        }

    def logit_scale(self) -> float:
        raise NotImplementedError

    def input_word_embed_dim(self) -> int:
        raise NotImplementedError

    def text_start_special_token_count(self) -> int:
        raise NotImplementedError

    def text_end_special_token_count(self) -> int:
        raise NotImplementedError

    def text_encoder_from_word_embeddings(self, input_word_embeds: torch.Tensor,
                                                      attn_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def load_model(self, path="model/frozenintime/frozenintime/configs/videocompare.json"):
        """
        Loads the model
        :param path:
        :return:
        """
        print("PATH IS:",
              path)  
        ckpt_save_dir = path[:path[:path.rfind("/")].rfind("/")]+"/saved" 
        print("CKPT SAVE DIR:",
              ckpt_save_dir)  #model/frozen-in-time/saved
        module_name = self.config['arch']['type']
        module_args = dict(self.config['arch']['args'])
        self.model = FrozenInTime(**module_args)
        pretrained = torch.load(self.config["pretrained"], map_location = DEVICE)
        state_dict = state_dict_data_parallel_fix(pretrained['state_dict'], self.model.state_dict())
        state_dict = self.model._inflate_positional_embeds(state_dict)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(DEVICE)
        return

    def tokenize(self, text):
        """
        Tokenizes text via tokenizer
        :param text: list of text to tokenize
        :return:
        """

        return self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)

    def text_encoder(self, text):
        """
        Encodes tokenized text into joint text/video embedding space
        :param text:
        :return:
        """
        return self.text_encoder_over_embeds(text)

    def get_input_word_embeddings(self, text_list: List[str]) -> torch.Tensor:
        """Converts a list of text string into a batched tensor of input word embeddings and a corresponding attention mask,
        including special tokens.

        Args:
            text_list (str): _description_

        Returns:
            torch.Tensor: input token embeddings for the text encoder. Shape (batch, sequence_len, token_dim)
            torch.Tensor: input sequence attention mask for the text encoder. Shape (batch, sequence_len)
        """
        text_tokens = self.tokenize(text_list)

        if self.cuda:
            text_tokens = text_tokens.to(DEVICE)

        text_input_embeds = self.model.compute_text(text_tokens)
        
        return text_input_embeds

    def text_encoder_over_embeds(self, text):
        with torch.no_grad():
            input_word_embeds = self.get_input_word_embeddings([text])
        return input_word_embeds.cpu()

    def open_video(self, video_path: str) -> np.ndarray:
        """
        Opens video and returns basic, non-transformed video tensor
        :param video:
        :return:
        """
        video_reader = decord.VideoReader(video_path, num_threads=1)
        return video_reader.get_batch(range(len(video_reader)))

    def transform(self, video, random_augment: bool = False):
        """
        Transforms video using model-specific transforms
        :param video:
        :return:
        """
        stride = self.config['data_loader']['args']['video_params']['stride']
        inputs = torch.permute(video, (0,3,2,1))
        inputs = torch.chunk(inputs, self.model.video_params.get('num_frames'))
        
        samples = []
        for i in [0,stride, 2*stride, self.model.video_params.get('num_frames')]:
            sample = []
            for chunk in inputs:
                sample.append(chunk[min(i,chunk.shape[0]-1)])
            sample = torch.stack(sample).float()
            sample = self.transforms(sample)
            
            f, c, w, h = sample.size()
            sample = sample.view(1,f,c,w,h)
            samples.append(sample)

        return torch.stack(samples).float()

    def video_encoder(self, video_path: str, subvideo_start_frame: Optional[int] = None, 
            subvideo_end_frame: Optional[int] = None, random_augment: bool = False) -> np.ndarray:
        """
        Load, transform and encode a video file into a joint text/video embedding space
        :param video:
        :return:
        """
        # Correct for any subvideo start/end frame information included in video_path ("{path}:{start}:{end}")
        video_path_split = video_path.split(":")
        if len(video_path_split) == 3:
            video_path = video_path_split[0]
        
        video = self.open_video(video_path)
        video_samples = self.transform(video)

        if self.cuda:
            video_samples = video_samples.to(DEVICE)

        video_features = []
        for sample in torch.unbind(video_samples):
            with torch.no_grad():
                video_feature = self.model.video_model.forward_features(sample)
            video_feature = self.model.video_model.head(video_feature)
            video_feature = self.model.vid_proj(video_feature)
            video_features.append(video_feature)
        
        video_features = torch.mean(torch.stack(video_features), dim=0)
        return video_features

    def default_similarity_metric(self) -> Similarity:
        """
        Returns a reference to the default similarity metric used by this VLM
        :return:
        """
        return sim_matrix

    def get_transforms(self):
        #Input is T, W, H, C
        transforms = Compose([
            Resize(256),
            CenterCrop(256),
            Resize(224),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        return transforms

    def get_train_transforms(self):
        # Input is T, H, W, C
        # Change to (T, C, H, W) for RandAugment
        transforms = Compose([
            Permute((0, 3, 1, 2)),
            RandAugment(magnitude=7, num_layers=4),
            Lambda(lambda x: x / 255.0),
            RandomResizedCrop(target_height=224, target_width=224, scale=(0.08, 1.0), aspect_ratio=(0.75, 1.3333)),
            RandomHorizontalFlip(p=0.5),
            # Change back to T, H, W, C
            Permute(dims=(1, 3, 0, 2)),

        ])

        return transforms

    def train(self, query_dataset: DatasetHandler, support_dataset: DatasetHandler, learning_rate, epochs, batch_size, n_way: int, n_support: int, n_query: Optional[int] = None, n_episodes: int = 1000, val_tuning_dataset: Optional[DatasetHandler] = None):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0)

        try:
            few_shot_dataset = FewShotTaskDataset(query_dataset, support_dataset, n_episodes, n_way, n_support, n_query,
                                                  val_tuning_dataset)
        except ValueError as e:
            # Skip invalid tests (if dataset too small, etc)
            print(e)
            return None
        
        for category_names, support_vid_paths, query_vid_paths, query_vid_labels, val_paths, val_labels in few_shot_dataset:
            train_dataloader = torch.utils.data.DataLoader(list(zip(query_vid_paths.flatten(), category_names.flatten())), batch_size = batch_size, num_workers=0, shuffle=True)
            
            for epoch_idx in range(epochs):
                for batch_idx, vids in enumerate(train_dataloader):
                    vid_paths = vids[0]
                    vid_labels = vids[1]
                    
                    optimizer.zero_grad()
                    
                    query_embeds = torch.cat([self.compute_video_embeds(vid_path) for vid_path in vid_paths])
                    text_embeds = torch.cat([torch.tensor(self.get_text_embeds(name), device=query_embeds.device) for name in vid_labels])

                    output = self.default_similarity_metric()(text_embeds, query_embeds)
                    
                    loss = self.loss(output)
                    loss.backward()
                    optimizer.step()
        return
