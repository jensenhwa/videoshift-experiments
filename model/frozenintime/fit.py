import numpy as np
import random
import os
from typing import Optional, List

import torch

from dataset.VideoReaderFromImages import VideoReaderFromImages
import model.model as module_arch
import json
from utils.util import state_dict_data_parallel_fix
import transformers

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from model.SimilarityVLM import SimilarityVLM
from similarity_metrics import Similarity

from torchvision.io import read_video
from pytorchvideo.transforms import *
from torchvision.transforms import Compose, Lambda, CenterCrop, RandomHorizontalFlip

import math
import decord
import pdb

# Default cache locations
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
CACHE_NAME = "cache"


class FiTVLM(SimilarityVLM):
    """
    Similarity-based VLM that uses Frozen-in-Time for frame and text encoders.
    """

    def __init__(self, path: str = "model/frozen-in-time/configs/videocompare.json",
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

        self.model = None
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

        # Do not load model, this is just dummy model to access methods
        if path is None:
            print("Dummy model loaded, no backbone or weights!")
            return

        assert type(self.path) is str
        assert type(self.num_seconds) is int

        # Load model
        self.load_model(path=self.path)

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

    def load_model(self, path="model/frozen-in-time/configs/videocompare.json"):
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
        self.model = getattr(module_arch, module_name)(**module_args)
        pretrained = torch.load(path)
        state_dict = state_dict_data_parallel_fix(pretrained['state_dict'], self.model.state_dict())
        self.model.load_state_dict(state_dict, strict=True)

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
            return input_word_embeds.cpu().numpy()

    def open_video(self, video_path: str) -> np.ndarray:
        """
        Opens video and returns basic, non-transformed video tensor
        :param video:
        :return:
        """
        if not (video_path.endswith(".mkv") or video_path.endswith(".mp4")):
            video_reader = VideoReaderFromImages(video_path, num_threads=1)
        else:
            video_reader = decord.VideoReader(video_path, num_threads=1)
        
        return video_reader.get_batch(range(len(video_reader)))

    def transform(self, video, random_augment: bool = False):
        """
        Transforms video using model-specific transforms
        :param video:
        :return:
        """
        if random_augment:
            inputs = self.train_transforms(video)
        else:
            inputs = self.transforms(video)
        # B, T, FPS, H, W, C (VideoCLIP is trained on 30 fps of s3d)
        _, h, w, c = inputs.size()
        inputs = inputs.view(1, -1, 30, h, w, c)  # Add singleton batch dimension
        return inputs

    def video_encoder(self, video_path: str) -> np.ndarray:
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
        #video = self.transform(video)

        if self.cuda:
            video = video.to(DEVICE)

        with torch.no_grad():
            video_features = self.model.compute_video(video)
            video_features = video_features.cpu().numpy()
        return video_features

    def default_similarity_metric(self) -> Similarity:
        """
        Returns a reference to the default similarity metric used by this VLM
        :return:
        """
        return module_arch.sim_matrix

    def get_transforms(self):
        # Input is T, H, W, C
        transforms = Compose([
            # Change to C, T, H, W for UniformTemporalSubsampling
            Permute((3, 0, 1, 2)),
            UniformTemporalSubsample(30*self.num_seconds),
            Lambda(lambda x: x / 255.0),  # Only normalization for VideoCLIP is / 255.0
            ShortSideScale(size=256),
            CenterCrop(224),
            # C, T, H, W -->, T, H, W, C
            Permute((1, 2, 3, 0)),
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
            Permute(dims=(0, 2, 3, 1)),

        ])

        return transforms
