
import os.path as osp

import torch
import torch.nn.functional as F

from tqdm import tqdm

from utils.metadata_tracker import MetadataTracker
from utils.openset_scores import csas, mls, msp


def compute_prototypes(feats, logits, n_classes, proto_weight_fn, n_clusters=8, n_run=32):
    """Compute prototypes based on image features (ideally extracted from a SSL ViT) and source modle logits predictions.

    Args:
        feats : image features (extracted from a SSL ViT ideally)
        logits : source model logits predictions
        n_classes : number of source modle classes
        proto_weight_fn : weight function to apply for computing prototypes.
        n_clusters : number of clusters to compute for the csas score. Defaults to 8.
        n_run (optional): number of Monte Carlo run for the csas score. Defaults to 32.

    Returns:
        class prototypes in the feature space
    """

    # Extract confident feature examples
    if proto_weight_fn == "CSAS":
        weights = csas(feats, logits, n_clusters, n_run=n_run)
        weights -= weights.min()
    elif proto_weight_fn == "MLS":
        weights = mls(logits)
        weights -= weights.min()
    elif proto_weight_fn == "MSP":
        weights = msp(logits)
    else:
        weights = torch.ones(feats.shape[0])
         
    preds = torch.argmax(logits, dim=-1)
    prototypes = torch.Tensor()

    for c in range(n_classes):
        # Compute mask for class c
        mask_c = (preds == c)
        weights_c = weights[mask_c]

        # Compute class weight
        weights_sum = weights.sum()

        # Compute prototype for class c
        proto = (feats[mask_c] * weights_c.unsqueeze(1)).sum(0) / weights_sum
        prototypes = torch.cat([prototypes, proto.unsqueeze(0)])

    return prototypes.numpy()


def compute_pseudo_logits(strong_features, logits, n_classes, proto_weight_fn = "CRMLS", n_clusters=8, n_run=32):
    """Compute pseudo logits based on prototypes estimated with strong_features, predicted logits and proto_weight_fn
    """
    prototypes = compute_prototypes(strong_features, logits, n_classes, n_clusters=n_clusters, n_run=n_run, proto_weight_fn=proto_weight_fn)
    pseudo_logits = torch.einsum("bd,nd->bn", F.normalize(strong_features, dim=-1), 
                                        F.normalize(torch.from_numpy(prototypes), dim=-1))    
    return pseudo_logits


@torch.no_grad()
def calibrate_bn_stats(net, train_loader, bn_epochs : int = 2):
    net.train()
    n_epochs = 0
    if bn_epochs > 0:
        print("\nCalibrating BN Stats !")

    # BN calibraton
    while(n_epochs < bn_epochs):
        pbar = tqdm(total=len(train_loader))

        for x, _, _ in train_loader:
            x = x.cuda()
            net(x)
            pbar.update(1)

        n_epochs += 1
        pbar.close()

    return net


@torch.no_grad()
def extract_features(net, experiment_dir : str, train_loader):
    """Function to extract features of a dataset given a trained neural network
    Features will be saved in the experiment_dir
    """
    # Set ViT features save file
    vit_feats_path = osp.join(experiment_dir, "vit_feats.pth")

    # If save file does not exists, we compute them and save them
    if not osp.exists(vit_feats_path):
        
        # Init metadata tracker
        metadata_tracker = MetadataTracker()
        net.eval()
        metadata_tracker.reset()

        # gather features
        for imgs, labels, indexes in tqdm(train_loader):

            imgs = imgs.cuda()

            # Compute Features form Vit
            feats = net(imgs)

            metadata_tracker.update_metadata({
                "label" : labels.cpu(),
                "index": indexes.cpu(),
                "vit_feats" : feats.cpu()
            })

        # Aggregate
        metadata_tracker.aggregate()

        # Save predictions
        metadata_tracker.to_pkl(["vit_feats"], experiment_dir)

    # Load feats and normalize them
    vit_feats = torch.load(vit_feats_path, map_location="cpu")
    vit_feats = F.normalize(vit_feats, dim=-1)
    vit_feats = F.normalize(vit_feats - vit_feats.mean(0), dim=-1)
    return vit_feats
