
import copy
import torch
import torchvision.models as models
import torch.nn as nn


class MobileNetV2(nn.Module):
    """MobileNetV2 with customizable output classes.

    Args:
        num_classes (int, optional): Number of output classes. Defaults to 10.
        dropout_p (float, optional): Dropout probability. Defaults to 0.0.

    Attributes:
        encoder (torch.nn.Module): MobileNetV2 feature encoder.
        fc (torch.nn.Module): Fully connected layer for classification.
        dim (int): Dimensionality of the classifier's input features.

    Methods:
        forward(x, return_feats=False): Forward pass through the network.
        resume_from_ckpt(ckpt_path=None): Load model weights from a checkpoint file.

    """

    def __init__(self, num_classes=10, dropout_p=0.0):
        """
        Initializes MobileNetV2.

        Args:
            num_classes (int, optional): Number of output classes. Defaults to 10.
            dropout_p (float, optional): Dropout probability. Defaults to 0.0.

        """
        super(MobileNetV2, self).__init__()
        self.encoder = models.__dict__['mobilenet_v2'](weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.fc = copy.deepcopy(self.encoder.classifier)
        self.dim = list(self.fc.children())[-1].in_features

        self.encoder.classifier = nn.Identity()

        if num_classes != 1000:
            self.fc = nn.Sequential(nn.Dropout(p=dropout_p, inplace=False),
                        nn.Linear(self.dim, num_classes))


    def forward(self, x, return_feats=False):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.
            return_feats (bool, optional): If True, returns intermediate features. Defaults to False.

        Returns:
            torch.Tensor or tuple: Model output tensor or tuple of output tensor and intermediate features.

        """
        feats = self.encoder(x)
        scores = self.fc(feats)

        if return_feats:
            return scores, feats

        return scores


    def resume_from_ckpt(self, ckpt_path : str = None):
        """
        Load model weights from a checkpoint file.

        Args:
            ckpt_path (str, optional): Path to the checkpoint file. Defaults to None.

        """

        if ckpt_path is not None:

            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt
            current_keys = self.state_dict()
            new_dict = {}

            for k, v in state_dict.items():

                if k not in current_keys:
                    k = k.replace("features", "encoder.features")
                    k = k.replace("classifier", "fc")
                    k = k.replace("ext.", "encoder.")

                new_dict[k] = v

            self.load_state_dict(new_dict, strict=True)
