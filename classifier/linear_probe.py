from typing import Optional
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression

from model.SimilarityVLM import SimilarityVLM
from .base import FewShotClassifier

'''
Simplest Linear Probe Classifier
'''

class LinearProbeFewShotClassifier(FewShotClassifier):
    def __init__(self, vlm: SimilarityVLM, regularization: float, lr: float = 1e-3, epochs=5, batch_size=2):
        self.vlm = vlm
        self.regularization = float(regularization)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        
    '''
    Returns a dict with the value of all classifier-specific parameters which may affect prediction
    accuracy (apart from the underlying VLM object used).
    This is used to differentiate test results which use different classifier parameters.
    '''
    def params(self) -> dict:
        return {
            "regularization": self.regularization
        }
        
    '''
    Predicts categories for a set of query videos in a few-shot task (formatted like FewShotTaskDataset)
    Args:
        category_names (np.array):          Array of names for each given few-shot category.
                                            Shape = (n_way,).
        support_video_paths (np.array):     Array of support video paths for each given few-shot category.
                                            Shape = (n_way, n_support).
                                            Can be None if n_support == 0.
        query_video_paths (np.array):       Array of query video paths to be predicted.
                                            Shape = (n_predict,).
        val_tuning_video_paths (Optional[np.array]):  Optional set of video paths from val split which the classifier can use to select the best-performing model/epoch.
        val_tuning_video_labels (Optional[np.array]): Labels for val_tuning_video_paths.
    Returns:
        (np.array):                         Predicted category index (with respect to the first index of the given
                                            category names and support videos) for each query video path.
                                            Shape = (n_predict,).
    '''
    def predict(self, category_names: np.ndarray, support_video_paths: Optional[np.ndarray], query_video_paths: np.ndarray, query_video_labels: np.ndarray,
                val_tuning_video_paths: Optional[np.array] = None, val_tuning_video_labels: Optional[np.array] = None) -> np.ndarray:
        assert query_video_labels is not None
        n_way = category_names.shape[0]
        n_predict = query_video_paths.shape[0]
        if support_video_paths is not None:
            n_support = support_video_paths.shape[1]
        else:
            n_support = 0
            
            
        # Use default similarity to text embeds if zero-shot
        if n_support == 0:

            train_dataloader = torch.utils.data.DataLoader(
                list(zip(query_video_paths.flatten(), category_names.flatten())),
                batch_size=self.batch_size, num_workers=0, shuffle=True
            )

            optimizer = torch.optim.Adam(self.vlm.model.parameters(), lr=self.lr, weight_decay=0)

            for epoch_idx in range(self.epochs):
                for batch_idx, (vid_paths, vid_labels) in enumerate(train_dataloader):
                    optimizer.zero_grad()

                    query_embeds = self.vlm.compute_video_embeds(vid_paths)
                    text_embeds = torch.tensor([self.vlm.get_text_embeds(name) for name in vid_labels], device=query_embeds.device)
                    print(query_embeds.shape, text_embeds.shape)
                    print(vid_paths)

                    logits_per_video = query_embeds @ text_embeds.t()
                    logits_per_text = text_embeds @ query_embeds.t()

                    targets = torch.arange(
                        query_embeds.size(0),
                        dtype=torch.long,
                        device=query_embeds.device)
                    loss_video = nn.CrossEntropyLoss()(logits_per_video, targets)
                    loss_text = nn.CrossEntropyLoss()(logits_per_text, targets)

                    loss = loss_video + loss_text

                    loss.backward()
                    optimizer.step()

            with torch.no_grad():
                query_embeds_torch = self.vlm.compute_video_embeds(query_video_paths)
                query_embeds = torch.tensor([self.vlm.get_video_embeds(vid) for vid in query_video_paths], device=query_embeds_torch.device)
                print(query_embeds, query_embeds_torch)
                print(query_embeds.shape, query_embeds_torch.shape)
                print(query_video_paths)
                print(torch.isclose(query_embeds, query_embeds_torch, atol=1e-04))
                text_embeds = torch.tensor([self.vlm.get_text_embeds(name) for name in category_names], device=query_embeds_torch.device)
                query_to_text_similarities = self.vlm.default_similarity_metric()(query_embeds, text_embeds)
                print(query_to_text_similarities)
                query_predictions = torch.argmax(query_to_text_similarities, dim=1)
                print(query_predictions, query_video_labels)
                accuracy = (query_predictions == torch.tensor(query_video_labels, device=query_predictions.device)).float().mean()
            return accuracy
        1/0
        # Linear probe ignoring text embeds
        query_embeds = np.array([self.vlm.get_video_embeds(vid) for vid in query_video_paths])
        support_embeds = np.array([self.vlm.get_video_embeds(vid) for vid in support_video_paths.flatten()])
        support_labels = np.repeat(np.arange(n_way), n_support)

        classifier = LogisticRegression(C = 1/self.regularization, max_iter=1000)
        classifier.fit(support_embeds, support_labels)
        
        query_predictions = classifier.predict(query_embeds)
        accuracy = (query_predictions == query_video_labels).mean()
        return accuracy