"""This script is inpired from https://github.com/agaldran/t3po"""

import os.path as osp

import torch.multiprocessing
import torchvision.transforms as transforms

from typing import Tuple

from .utils import subsample_classes, IndexedConcatDataset, IndexedImageFolder

torch.multiprocessing.set_sharing_strategy('file_system')


osr_splits = {
    # ['01_TUMOR', '02_STROMA',  '03_COMPLEX', '04_LYMPHO', '05_DEBRIS',  '06_MUCOSA', '07_ADIPOSE', '08_EMPTY']
    'kather16': {

        # Number of classes
        "n_classes": 8,

        # Source-Free Open Set Domain Adaptation splits
        "splits": [
            [0, 1,    3,    5    ],    # S1: TUMOR, STROMA, LYMPHO and MUCOSA
            [0, 1,               ],    # S2: TUMOR and STROMA     
            [0, 1,    3          ],    # S3: TUMOR, STROMA and LYMPHO
        ]
    },

    ####################################################################################################################
    # KATHER_100K

    'kather19': {

        # Number of classes
        "n_classes": 9,
    
        # Source-Free Open Set Domain Adaptation splits
        # We map classnames with respect to Kather2016 dictionnary
        "splits": [
            [8, (5,7), 3, 6],          # S1: TUM, (MUS,STR), LYM and NORM -> TUMOR, STROMA, LYMPHO and MUCOSA
            [8, (5,7)],                # S2: TUM and (MUS,STR)            -> TUMOR and STROMA
            [8, (5,7), 3],             # S3: TUM, (MUS,STR) and LYM       -> TUMOR, STROMA and LYMPHO
        ]
    },

    ####################################################################################################################
    # CRCTP

    'crctp': {
        
        # Number of classes
        "n_classes": 7,

        # Source-Free Open Set Domain Adaptation splits
        # We map classnames with respect to Kather2016 dictionnary
        "splits": [
            [(6,1), (4,5), 3, 0],          # S1: (Tumor, Complex Stroma), (Stroma,Musle), Inflammatory and Benign -> TUMOR, STROMA, LYMPHO and MUCOSA
            [(6,1), (4,5)],          # S2: (Tumor, Complex Stroma) and (Stroma,Musle) -> TUMOR and STROMA
            [(6,1), (4,5), 3],          # S1: (Tumor, Complex Stroma), (Stroma,Musle) and Inflammatory -> TUMOR, STROMA and LYMPHO

        ]
    }
}


def get_histopathology_transform(image_size=224):
    """Generate image transformation functions for histology images.

    Args:
        image_size (int, optional): Target image size. Defaults to 224.

    Returns:
        tuple: Transformation functions for training and testing.
    """

    # Mean and standard deviation values for normalization
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    # Training transformation pipeline
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

     # Testing transformation pipeline
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return (train_transform, test_transform)


def load_source_datasets(root_dir : str, train_transform, test_transform, known_classes : Tuple[int],
                                    unknown_classes : Tuple[int], return_idx : bool=False):
    """Load source dataset splits for training, validation, and testing.

    Args:
        root_dir (str): Path to the dataset folder.
        train_transform (torchvision.transforms): Transform function for training images.
        test_transform (torchvision.transforms): Transform function for test images.
        known_classes (Tuple): Tuple of index classes considered as known.
        unknown_classes (Tuple): Tuple of index classes considered as unknown.
        return_idx (bool, optional): Whether the dataset returns index when an image is requested. Defaults to False.

    Returns:
        dict: Dictionary containing dataset splits.
    """

    # Set split paths
    train_dir = osp.join(root_dir, 'train')
    val_dir = osp.join(root_dir, 'val')
    test_dir = osp.join(root_dir, 'test')

    # Build train set
    train_dataset = IndexedImageFolder(root=train_dir, transform=train_transform, return_idx=return_idx)
    train_dataset = subsample_classes(train_dataset, include_classes=known_classes)
    print("Training Known Dataset size : ", len(train_dataset))

    # Build validation sets
    val_dataset_known = IndexedImageFolder(root=val_dir, transform=test_transform, return_idx=return_idx)
    val_dataset_known = subsample_classes(val_dataset_known, include_classes=known_classes)
    print("Validation Known Dataset size : ", len(val_dataset_known))

    val_dataset_unknown = IndexedImageFolder(root=val_dir, transform=test_transform, return_idx=return_idx)
    val_dataset_unknown = subsample_classes(val_dataset_unknown, include_classes=unknown_classes, is_ood=True)
    print("Validation Unknown Dataset size :", len(val_dataset_unknown))

    # Build test sets
    test_dataset_known = IndexedImageFolder(root=test_dir, transform=test_transform, return_idx=return_idx)
    test_dataset_known = subsample_classes(test_dataset_known, include_classes=known_classes)
    print("Test Known Dataset size : ", len(test_dataset_known))

    test_dataset_unknown = IndexedImageFolder(root=test_dir, transform=test_transform, return_idx=return_idx)
    test_dataset_unknown = subsample_classes(test_dataset_unknown, include_classes=unknown_classes, is_ood=True)
    print("Test Unknown Dataset size :", len(test_dataset_unknown))

    test_dataset = IndexedConcatDataset([test_dataset_known, test_dataset_unknown])

    # Construct dictionary of all datasets
    all_datasets = {
        'train': train_dataset,
        'val_known': val_dataset_known,
        'val_unknown': val_dataset_unknown,
        'val' : IndexedConcatDataset([val_dataset_known, val_dataset_unknown]),
        'test_known' : test_dataset_known,
        'test_unknown' : test_dataset_unknown,
        'test': test_dataset,
    }

    return all_datasets


def load_target_datasets(root_dir : str, test_transform, known_classes : Tuple[int],
                            unknown_classes : Tuple[int], return_idx : bool=False):
    """Load target dataset splits for training and testing.

    Args:
        root_dir (str): Path to the dataset folder.
        test_transform (torchvision.transforms): Transform function for test images.
        known_classes (Tuple): Tuple of index classes considered as known.
        unknown_classes (Tuple): Tuple of index classes considered as unknown.
        return_idx (bool, optional): Whether the dataset returns index when an image is requested. Defaults to False.

    Returns:
        dict: Dictionary containing dataset splits.
    """

    # Set split paths
    train_dir = osp.join(root_dir, 'train')
    test_dir = osp.join(root_dir, 'test')

    # Build train set (Unlabeled)
    train_dataset_known = IndexedImageFolder(root=train_dir, transform=test_transform, return_idx=return_idx)
    train_dataset_known = subsample_classes(train_dataset_known, include_classes=known_classes)
    print("Train Known Dataset size : ", len(train_dataset_known))

    train_dataset_unknown = IndexedImageFolder(root=train_dir, transform=test_transform, return_idx=return_idx)
    train_dataset_unknown = subsample_classes(train_dataset_unknown, include_classes=unknown_classes, is_ood=True)
    print("Train Unknown Dataset size : ", len(train_dataset_unknown))

    # Build test dataset and subsample known/unknown classes
    test_dataset_known = IndexedImageFolder(root=test_dir, transform=test_transform, return_idx=return_idx)
    test_dataset_known = subsample_classes(test_dataset_known, include_classes=known_classes)
    print("Test Known Dataset size : ", len(test_dataset_known))

    test_dataset_unknown = IndexedImageFolder(root=test_dir, transform=test_transform, return_idx=return_idx)
    test_dataset_unknown = subsample_classes(test_dataset_unknown, include_classes=unknown_classes, is_ood=True)
    print("Test Unknown Dataset size :", len(test_dataset_unknown))

    # Construct dictionary of all datasets
    all_datasets = {
        'train' : IndexedConcatDataset([train_dataset_known, train_dataset_unknown]),
        'test_unknown': test_dataset_unknown,
        'test_known': test_dataset_known,
        'test': IndexedConcatDataset([test_dataset_known, test_dataset_unknown])
    }

    return all_datasets


def get_histopathology_datasets(root_dir : str, name : str, split_idx : int, transform = None,
                                image_size : int = 150, target : bool = True, return_idx : bool = True):
    """Load histopathology dataset splits.

    Args:
        root_dir (str): Path to the dataset folder.
        name (str): Dataset name.
        split_idx (int): Index of the split to load.
        transform: Predefined transform function to apply to all splits.
        image_size (int, optional): Size of the image. Defaults to 150.
        target (bool, optional): Whether to load target dataset splits or source dataset splits. Defaults to True.
        return_idx (bool, optional): Whether the dataset returns index when an image is requested. Defaults to True.

    Returns:
        dict: Dataset splits and the number of classes.
    """

    # Set known and unknown class indexes
    real_name = "kather2019" if "kather2019" in name else name
    print('\nLoading dataset {}'.format(real_name))

    dataset_info = osr_splits[real_name]
    root_dir = osp.join(root_dir, name)
    max_n_classes = dataset_info["n_classes"]
    known_classes = dataset_info["splits"][split_idx-1]
    n_classes = len(known_classes)
    known_classes_f = []
    for classes in known_classes:
        if isinstance(classes, tuple):
            known_classes_f += list(classes)
        else:
            known_classes_f.append(classes)

    unknown_classes = [x for x in range(max_n_classes) if x not in known_classes_f]
    print(f'{name} known classes: {known_classes}')
    print(f'{name} open set classes: {unknown_classes}')

    # Set transform functions
    if transform is not None:
        train_transform = test_transform = transform
    else:
        train_transform, test_transform = get_histopathology_transform(image_size=image_size)

    # Load dataset splits
    if target:
        datasets = load_target_datasets(root_dir, test_transform, known_classes, 
                                            unknown_classes, return_idx=return_idx)
    else:
        datasets = load_source_datasets(root_dir, train_transform, test_transform,
                                            known_classes, unknown_classes, return_idx=return_idx)

    return datasets, n_classes
