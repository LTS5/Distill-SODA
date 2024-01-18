
import argparse

from torch.utils.data import DataLoader

from methods import DistillSODA
from datasets import get_datasets
from networks import get_network
from utils.utils import setup_experiment, seed_worker


def get_opt_parser():
    
    parser = argparse.ArgumentParser(description="Distill-SODA: Source-Free Open-Set Domain Adaptation in Computational Pathology", add_help=False)

    # File Paths
    parser.add_argument("--data_path", type=str, help="Path to the target dataset")
    parser.add_argument("--source_model_path", type=str, help="Path to the source model weights")
    parser.add_argument("--sslvit_model_path", type=str, help="Path to the self-supervised ViT weights")
    parser.add_argument("--experiment_dir", type=str, help="Path to the experiment folder")
    parser.add_argument("--save_every", type=int, default=5, help="Frequency at which the target model weights are saved")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Dataset Configuration
    parser.add_argument("--dataset_name", type=str, default="kather16", choices=("kather16", "kather19"),
                        help="Name of the dataset, corresponds to the folder name of the dataset")
    parser.add_argument("--split_idx", type=int, default=1, choices=(1, 2, 3),
                        help="Split index to use, see different splits in datasets/histo.py")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Image size; 224 for kather-19, 150 for kather-16 and CRCTP")

    # Model Architecture
    parser.add_argument("--source_model_arch", type=str, default="mobilenet_v2", choices=("mobilenet_v2",),
                        help="Source model architecture")
    parser.add_argument("--sslvit_model_arch", type=str, default="dino_vit_base", choices=("dino_vit_base",),
                        help="Self-supervised ViT model architecture")
    
    # Optimization Parameters
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--bn_epochs", type=int, default=2, help="Epochs to calibrate BN layers")
    parser.add_argument("--lr", type=float, default=1e-3)

    # Hyperparameters
    parser.add_argument("--temp", type=float, default=0.07, help="Temperature for scaling pseudo logits in Distill-SODA")
    parser.add_argument("--n_kmeans", type=int, default=32, help="Number of Monte-Carlo runs for computing CSAS score")
    parser.add_argument("--n_clusters", type=int, default=16, help="Number of clusters for K-Means clustering for CSAS score")
    parser.add_argument("--proto_weight_fn", type=str, default="CSAS", choices=("MSP", "MLS", "CSAS", "UNIFORM"))

    # Distributed Training Parameters
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--dist_url", type=str, default="env://")
    parser.add_argument("--port", type=str, default="29500")

    return parser


def run(opt):

    # Set up the experiment
    setup_experiment(opt, seed=opt.seed, experiment_dir=opt.experiment_dir)

    # Load datasets for training and testing
    datasets, n_classes = get_datasets(root_dir=opt.data_path, name=opt.dataset_name, split_idx=opt.split_idx,
                             image_size=opt.image_size, target=True, return_idx=True)
    test_dataset = datasets["test"]
    train_dataset = datasets["train"]

    # Set data loaders
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size*4, shuffle=False, 
                        drop_last=False, num_workers=opt.num_workers, worker_init_fn=seed_worker)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                        drop_last=False, num_workers=opt.num_workers, worker_init_fn=seed_worker)
    train_loader_eval = DataLoader(train_dataset, batch_size=opt.batch_size*4, shuffle=False,
                        drop_last=False, num_workers=opt.num_workers, worker_init_fn=seed_worker) 

    # Get source model and self-supervised ViT model
    source_model = get_network(opt.source_model_arch, n_classes=n_classes, ckpt_path=opt.source_model_path).cuda()
    sslvit_model = get_network(opt.sslvit_model_arch, ckpt_path=opt.sslvit_model_path).cuda()

    # Set up the Trainer
    trainer = DistillSODA(source_model=source_model, ssl_vit_model=sslvit_model, experiment_dir=opt.experiment_dir,
                    epochs=opt.epochs, bn_epochs=opt.bn_epochs, n_classes=n_classes, lr=opt.lr,
                    n_clusters=opt.n_clusters, n_kmeans=opt.n_kmeans, temp=opt.temp, proto_weight_fn=opt.proto_weight_fn)

    # Train and Test
    trainer.train(train_loader=train_loader, train_loader_eval=train_loader_eval, test_loader=test_loader, save_every=opt.save_every)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Proto-SFOSDA Main', parents=[get_opt_parser()])
    opt = parser.parse_args()
    run(opt)
