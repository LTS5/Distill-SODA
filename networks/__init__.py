
from .mobilenet import MobileNetV2
from .vit import DINOViT, vit_base


def get_network(network_name, n_classes=10, ckpt_path=None, dropout_p=0.0):
    """Get a pre-trained neural network for classification.

    Args:
        network_name (str): Name of the desired neural network.
        n_classes (int, optionnal): Number of output classes. Default to 10
        ckpt_path (str, optional): Path to the checkpoint file. Defaults to None.
        dropout_p (float, optional): Dropout probability. Defaults to 0.0.

    Returns:
        torch.nn.Module: Pre-trained neural network.

    Raises:
        NotImplementedError: If the specified network is not implemented.

    """
    if network_name == "mobilenet_v2":
        net = MobileNetV2(n_classes, dropout_p=dropout_p)
        net.resume_from_ckpt(ckpt_path)
    elif network_name == "dino_vit_base":
        net = DINOViT(vit_base())
        net.resume_from_ckpt(ckpt_path)
    else:
        raise NotImplementedError

    return net
