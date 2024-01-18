
import numpy as np
import torchvision

from typing import List
from PIL import Image

from torch.utils.data import ConcatDataset


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            my_img = img.convert('RGB')
        return my_img


class IndexedImageFolder(torchvision.datasets.ImageFolder):
    """ImageFolder with indexes
    """

    def __init__(self, root : str, transform, return_idx : bool):
        """Initialize the dataset

        Args:
            root (str): path to the dataset
            transform (torchvision.transforms): transform function to apply to images
            return_idx (bool): whether the dataset return index when an image is requested.
        """
        super(IndexedImageFolder, self).__init__(root, transform)
        self.return_idx = return_idx


    def __getitem__(self, item):
        img, label = super().__getitem__(item)

        if self.return_idx:
            return img, label, item
        return img, label



class IndexedConcatDataset(ConcatDataset):
    """ConcatDataset Extension for IndexedImageFolder objects
    """

    def __init__(self, datasets : List[IndexedImageFolder]) -> None:
        """Initialize the dataste with a list of IndexedImageFolder to concatenate

        Args:
            datasets (List[IndexedImageFolder]): a list of IndexedImageFolder
        """
        super().__init__(datasets)
        self.return_idx = datasets[0].return_idx


    def __getitem__(self, idx):

        if self.return_idx:
            img, label, _ = super().__getitem__(idx)
            return img, label, idx
        else:
            return super().__getitem__(idx)


def subsample_dataset(dataset, idxs):
    # dataset.imgs = [x for i, x in enumerate(dataset.imgs) if i in idxs]
    # dataset.samples = [x for i, x in enumerate(dataset.samples) if i in idxs]
    # I am replacing the above awful code by a faster version, no need to enumerate for each idx the same set again and
    # again, people should be more careful with comprehensions (agaldran).
    imgs, sampls = [], []
    for i in idxs:
        imgs.append(dataset.imgs[i])
        sampls.append(dataset.samples[i])
    dataset.imgs = imgs
    dataset.samples = sampls
    dataset.targets = np.array(dataset.targets)[idxs].tolist()

    return dataset


def subsample_classes(dataset, include_classes=(0,1), is_ood=False):
    cls_idxs = []

    target_xform_dict = {}
    i = -1
    for k in include_classes:

        if not is_ood:
            i += 1

        if isinstance(k, tuple):
            for l in k:
                target_xform_dict[l] = i
                cls_idxs += [x for x,y in enumerate(dataset.targets) if y == l]
        else:
            target_xform_dict[k] = i
            cls_idxs += [x for x,y in enumerate(dataset.targets) if y == k]
      

    dataset = subsample_dataset(dataset, cls_idxs)
    dataset.target_transform = lambda x: target_xform_dict[x]

    # torchvision ImageFolder dataset have a handy class_to_idx attribute that we have spoiled and need to re-do
    # filter class_to_idx to keep only include_classes
    new_class_to_idx = {key: val for key, val in dataset.class_to_idx.items() if val in target_xform_dict.keys()}
    # fix targets so that they start in 0 and are correlative
    new_class_to_idx = {k: target_xform_dict[v] for k, v in new_class_to_idx.items()}
    # replace bad class_to_idx with good one
    dataset.class_to_idx = new_class_to_idx

    # and let us also add a idx_to_class attribute
    dataset.idx_to_class = dict((v, k) for k, v in new_class_to_idx.items())

    return dataset
