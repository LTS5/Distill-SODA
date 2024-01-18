
import os
import torch
import torch.nn as nn

from tqdm import tqdm

from .utils import compute_pseudo_logits, calibrate_bn_stats, extract_features
from utils.metadata_tracker import MetadataTracker
from utils.metric_tracker import MetricTracker
from utils.metrics import accuracy, auroc
from utils.openset_scores import mls
from utils.utils import ensure_dir


class DistillSODA:
    """
    Class for Distill-SODA.

    Args:
        source_model (nn.Module): Source model for the task.
        ssl_vit_model (str): SSL ViT model for the task.
        experiment_dir (str): Directory to store experiment results.
        epochs (int, optional): Number of training epochs. Defaults to 10.
        bn_epochs (int, optional): Number of epochs to calibrate Batch Normalization statistics. Defaults to 2.
        n_classes (int, optional): Number of classes in the source model. Defaults to 10.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        n_clusters (int, optional): Number of clusters to compute for the CSAS score. Defaults to 10.
        proto_weight_fn (str, optional): Prototype weight function name. Defaults to "CSAS".
        n_kmeans (int, optional): Number of k-means runs for the CSAS score. Defaults to 32.
        temp (float, optional): Temperature parameter for distillation. Defaults to 1e-3.
    """

    def __init__(self, source_model, ssl_vit_model : str, experiment_dir : str, epochs : int=10, bn_epochs : int = 2, 
                    n_classes : int = 10, lr : float = 1e-3, n_clusters : int = 10, proto_weight_fn : str = "CSAS",
                    n_kmeans : int = 32, temp : float = 1e-3) -> None:          

        # Set attributes
        self.source_model = source_model
        self.ssl_vit_model = ssl_vit_model
        self.experiment_dir = experiment_dir
        self.epochs = epochs
        self.bn_epochs = bn_epochs
        self.n_classes = n_classes
        self.lr = lr
        self.n_clusters = n_clusters
        self.proto_weight_fn = proto_weight_fn
        self.n_kmeans=n_kmeans
        self.temp=temp

        # Optimizer
        self.optimizer = torch.optim.Adam(self.source_model.parameters(), lr=lr)

        # Distillation Criterion
        self.rec_criterion = nn.MSELoss()

        # Setup trackers
        self.metadata_tracker = MetadataTracker()
        self.metric_tracker = MetricTracker()
        self.cur_epoch = 0


    @torch.no_grad()
    def predict(self, loader):
        """
        Predicts using the source model.

        Args:
            loader: DataLoader for prediction.

        Returns:
            Tuple: Tuple containing logits and labels.
        """
        self.source_model.eval()
        all_logits = torch.empty((len(loader.dataset), self.n_classes))
        all_labels = torch.empty((len(loader.dataset),)).to(torch.long)

        for imgs, labels, indexes in loader:
            imgs = imgs.cuda()
            logits = self.source_model(imgs)
            all_logits[indexes] = logits.cpu()
            all_labels[indexes] = labels.cpu()

        return all_logits, all_labels


    @torch.no_grad()
    def test(self, test_loader):
        """
        Tests the source model on the given test loader.

        Args:
            test_loader: DataLoader for testing.
        """
        # Set model in eval mode
        self.source_model.eval()
        self.metadata_tracker.reset()

        # Predict
        for img, label, _ in tqdm(test_loader):

            # Get Prediction
            logit, feat = self.source_model(img.cuda(), return_feats=True)
            pred = torch.argmax(logit, dim=-1)

            self.metadata_tracker.update_metadata({
                "label" : label.cpu(),
                "pred" : pred.cpu(),
                "logit" : logit.cpu(),
                "feat" : feat.cpu()
            })

        # Aggregate
        self.metadata_tracker.aggregate()

        # Save predictions
        target_dir = os.path.join(self.experiment_dir, f"Test_Evaluation_{self.cur_epoch}")
        ensure_dir(target_dir)
        self.metadata_tracker.to_csv(["label", "pred"], target_dir)
        self.metadata_tracker.to_pkl(["logit", "feat"], target_dir)

        # Log metrics
        self.log_metrics(self.metadata_tracker["label"] ,self.metadata_tracker["logit"])


    def log_metrics(self, labels, logits):
        """
        Computes and logs metrics.

        Args:
            labels: Ground truth labels.
            logits: Model logits.
        """
        mask_id = labels >= 0
        preds = torch.argmax(logits, dim=-1)

        # Accuracy
        accuracy_ = accuracy(preds.numpy()[mask_id], labels.numpy()[mask_id])
        print(f"Accuracy: %4.4f" % (accuracy_))

        # OoD Metrics
        roc_score = mls(logits)
        auroc_, _, _, _, _ = auroc(roc_score, mask_id)

        # Log
        print(f"AUROC : %4.4f\n" % (auroc_))


    def train(self, train_loader, train_loader_eval, test_loader, save_every : int=5):
        """
        Train the DistillSODA model.

        Args:
            train_loader: DataLoader for training.
            train_loader_eval: Dataloader for evaluating training data.
            test_loader: DataLoader for testing.
            save_every (int, optional): Save model every specified number of epochs. Defaults to 5.
        """

        # Extract ViT features
        print("\nComputing target image features in the SSL ViT feature space !")
        vit_feats = extract_features(self.ssl_vit_model, self.experiment_dir, train_loader_eval)

        # Calibrate BN Stats and Load training logits
        print("Calibrating source model !")
        self.source_model = calibrate_bn_stats(self.source_model, train_loader, self.bn_epochs)

        # Computing target image predictions using the source model
        print("\nComputing calibrated model predictions on the target data !")
        all_logits, all_labels = self.predict(train_loader_eval)
        self.log_metrics(all_labels, all_logits)

        # Compute and Save Pseudo Logits
        with torch.no_grad():
            print(f"\nComputing pseudo labels !")
            pseudo_logits = compute_pseudo_logits(vit_feats, all_logits, self.n_classes, 
                                                    n_clusters=self.n_clusters, n_run=self.n_kmeans,
                                                    proto_weight_fn=self.proto_weight_fn)
            pseudo_logits /= self.temp

            # Log Metrics
            self.log_metrics(all_labels, pseudo_logits)

            # Save Pseudo Labels
            self.metadata_tracker.reset()
            self.metadata_tracker.update_metadata({
                "label" : all_labels.cpu(),
                "pred" : torch.argmax(pseudo_logits, dim=-1).cpu(),
                "logit" : pseudo_logits.cpu()
            })

            self.metadata_tracker.aggregate()
            target_dir = os.path.join(self.experiment_dir, "Pseudo_Predictions")
            ensure_dir(target_dir)
            self.metadata_tracker.to_csv(["label", "pred"], target_dir)
            self.metadata_tracker.to_pkl(["logit"], target_dir)


        # Train
        for e in range(self.epochs):

            # Setup Epoch
            self.cur_epoch = e+1
            pbar = tqdm(total=len(train_loader))
            self.source_model.train()
            self.metric_tracker.reset()

            # Train One Epoch
            for img, _, idx in train_loader:

                # Load input and target
                img = img.cuda()

                # Rec loss
                self.optimizer.zero_grad()
                logit = self.source_model(img)
                rec_loss = self.rec_criterion(pseudo_logits[idx].cuda(), logit)
                rec_loss.backward()

                # Backward
                self.optimizer.step()

                # Log
                self.metric_tracker.update_metrics({
                    "Rec_Loss" : rec_loss.item(),
                }, batch_size=len(img))

                # Log Metrics
                pbar.set_description(self.metric_tracker.log(prefix="Epoch {:d}".format(self.cur_epoch)))
                pbar.update()

            # Test and save model
            pbar.close()

            print("Test on the target test set!")
            self.test(test_loader)

            if self.cur_epoch % save_every == 0:
                torch.save(self.source_model.state_dict(), os.path.join(self.experiment_dir, f"Net_Epoch[{self.cur_epoch}].pth"))

            # Save the model
            torch.save(self.source_model.state_dict(), os.path.join(self.experiment_dir, "net.pth"))
