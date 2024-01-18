
from .histo import get_histopathology_datasets


def get_datasets(root_dir : str, name : str, split_idx : int, transform = None,
                    image_size : int = 224, target : bool = False, return_idx : bool = False):
    """Load dataset splits given a path and a dataset name.

    Args:
        root_dir (str): Path to the dataset folder.
        name (str): Dataset name.
        split_idx (int): Index of the split to load.
        transform: Predefined transform function to apply to all splits.
        image_size (int, optional): Size of the image. Defaults to 224.
        target (bool, optional): Whether the dataset is loaded as a target. Defaults to False.
        return_idx (bool, optional): Whether the dataset returns the index of the requested image. Defaults to False.

    Returns:
        dict, int: Dataset splits, number of classes
    """
    if name in ["kather16", "kather19", "crctp"]:
        datasets, n_classes = get_histopathology_datasets(root_dir, name, split_idx, transform=transform,
                                                image_size=image_size, target=target, return_idx=return_idx)
    else:
        raise NotImplementedError

    return datasets, n_classes
