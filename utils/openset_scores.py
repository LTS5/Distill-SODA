
import torch
import faiss

from tqdm import tqdm


def msp(logits : torch.Tensor):
    """Compute maximum softmax propability (MSP) score
    """
    return torch.amax(torch.softmax(logits, dim=-1), dim=-1)


def mls(logits: torch.Tensor):
    """Compute maximum logit score (MLS)
    """
    return torch.amax(logits, dim=-1)


def csas(strong_features, logits, n_clusters, n_iter=20, n_run=4):
    """Computes slosed affinity score
    """

    # Compute mls score for every example
    mls_ = mls(logits)
    crlcs_ = torch.zeros_like(mls_)

    # Setup Clustering with FAISS
    verbose=False
    bs, d = strong_features.shape
    
    # Run K-Means clustering n_run times
    for seed in tqdm(range(n_run)):

        # Run K-Means
        random_indices = torch.randperm(bs)
        kmeans = faiss.Kmeans(d, n_clusters, niter=n_iter, verbose=verbose, gpu=False, spherical=True,
                                seed=seed, nredo=10, min_points_per_centroid=100)
        kmeans.train(strong_features[random_indices])
        D, predictions = kmeans.index.search(strong_features, 1)
        predictions = predictions.squeeze()

        # Compute CSAS for every cluster
        for i in range(n_clusters):
            mask = predictions == i
            cluster_size = sum(mask)
            c_logits = logits[mask].mean(0)
            crlcs_[mask] += torch.amax(c_logits)/cluster_size

    crlcs_ /= n_run
    return crlcs_
