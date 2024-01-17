# Distill-SODA

This repository contains official [PyTorch](https://pytorch.org/) implementation for paper **Distill-SODA: Distilling Self-Supervised Vision Transformer for Source-Free Open-Set Domain Adaptation in Computational Pathology** by Guillaume Vray, Devavrat Tomar, Behzad Bozorgtabar, and Jean-Philippe Thiran.\
[[arxiv]](https://arxiv.org/abs/2307.04596)

<img src="image/distillsoda.png" alt="drawing" width="1000"/>

**Unveiling the self-supervised vision transformer (ViT) for source-free open-set domain adaptation (SF-OSDA).** The source model $f_s$ undergoes adaptation, resulting in the adapted model $f_t$ acclimating to an unlabeled target domain, accommodating both closed-set (known) and open-set (unknown) classes, all while maintaining a strict boundary of not accessing the source dataset. Our methodology capitalizes on distilling knowledge from a self-supervised ViT, leveraging its potent capability to generate contextually enriched target embeddings. Guidance for knowledge distillation can originate from two principal sources: self-supervised pre-trained transformer models without adaptation and models that have undergone extensive self-supervised pre-training on publicly available histopathology datasets or target domain data, showcasing our approach's adaptability.


## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)


## Installation

Get started with this project by setting it up on your local machine. Follow these steps:

#### Clone the Repository
```bash
git clone git@github.com:LTS5/Distill-SODA.git
cd Distill-SODA
```

#### Install Dependencies
```bash
conda create --name DistillSODA python=3.8
conda activate SODA

pip install torch==1.13.0 torchvision==0.14.0 --extra-index-url https://download.pytorch.org/whl/cu113
conda install -c pytorch faiss-gpu cudatoolkit=11.3
pip install kornia==0.6.8
pip install tqdm
pip install pandas
pip install scikit-learn
```

## Usage

### (A) Data Preparation

To reproduce our results, download the CRC tissue characterization datasets used in our work:

+ Kather-16:
```bash
# Download
wget https://zenodo.org/record/53169/files/Kather_texture_2016_image_tiles_5000.zip
unzip Kather_texture_2016_image_tiles_5000.zip -d data/

# Split
python data/kather16.py --path_data_in data/Kather_texture_2016_image_tiles_5000           

# Clean
rm -r data/Kather_texture_2016_image_tiles_5000
rm Kather_texture_2016_image_tiles_5000.zip
```

+ Kather-19:
```bash
# Download
wget https://zenodo.org/record/1214456/files/NCT-CRC-HE-100K.zip
unzip NCT-CRC-HE-100K.zip -d data/

# Split
python data/kather19.py --path_data_in data/NCT-CRC-HE-100K           

# Clean
rm -r data/NCT-CRC-HE-100K
rm NCT-CRC-HE-100K.zip
```

+ CRCTP:
```
!! The CRCTP is not publicly available anymore !!
```


### (B) Download Source Model

Download our source models following the google drive links below:

| Dataset | Open-Set Detection Method | Backbone | Download |
| --------|---------------------------|--------------|----------|
|Kather-16| CE+ | MobileNet-V2 | [ckpt](https://drive.google.com/drive/folders/1wApBoYhNW0V5mSjpfBVMNdZCzyU3hRPe?usp=drive_link) |
|Kather-19| CE+ | MobileNet-V2 | [ckpt](https://drive.google.com/drive/folders/1bghiqoZV9M4WVpD-ff-S-LEaGNHPUe6L?usp=drive_link) |
|CRCTP    | CE+ | MobileNet-V2 | [ckpt](https://drive.google.com/drive/folders/1eoJKl9cKC_s1-nqRH4ejCsgHT4Z588rk?usp=drive_link) |

If you wish to re-train or train models on your source dataset, use the [T3PO repo](https://github.com/agaldran/t3po) following their instructions.

### (C) Run Distill-SODA

Adapt the source model on the target data by distilling knowledge from a self-supervised ViT:

```bash
python main_distillsoda.py --data_path path/to/target_data \
        --experiment_dir path/to/saving_dir \
        --dataset_name dataset_name --split_idx split_idx --seed seed \
        --source_model_path /path/to/source_model \
        --sslvit_model_path /path/to/ssl_vit_model
```

Our method seamlessly integrates with popular public models, such as DINO (ViT-B/16). If you wish to leverage these models, you can easily copy the [weights](https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain_full_checkpoint.pth) to use them as a starting point for your experiments.


### (D) Run DINO+AdvStyle

In our work, we introduced DINO+AdvStyle, a novel style-based adversarial data augmentation. This technique serves as hard positives for self-training a ViT, leading to highly contextualized embeddings for the target dataset. The incorporation of DINO+AdvStyle significantly enhances the performance of Distill-SODA.
To leverage DINO+AdvStyle and self-train a pretrained ViT on the target dataset, run the following script:

```bash
python -m torch.distributed.launch --nproc_per_node=2 main_dinoadvstyle.py --arch vit_base --data_path path/to/target_data \
        --dataset_name dataset_name --output_dir path/to/saving_dir \
        --momentum_teacher 0.996 --warmup_epochs 0 --batch_size_per_gpu 16
```

This command initiates the self-training process, utilizing DINO+AdvStyle for robust and context-aware embeddings. The resulting pretrained ViT can be further employed in the Distill-SODA process in (See (C)), enhancing the adaptation on the target dataset.


To reproduce our results, you can find the self-trained ViT checkpoints for the three utilized datasets below:

| Dataset | SSL Backbone | Download |
| --------|--------------|----------|
|Kather-16| ViT-B/16 | [ckpt](https://drive.google.com/drive/folders/1LS4MXO6BzaR1A31-1E8mzxFgAwwff4-B?usp=drive_link) |
|Kather-19| ViT-B/16 | [ckpt](https://drive.google.com/drive/folders/14Lpr_nJtsckxSwfZhTZF-UgJry-Aq8u4?usp=drive_link) |
|CRCTP    | ViT-B/16 | [ckpt](https://drive.google.com/drive/folders/11eXm09z876NBqgbfBNKyypVpBIn_b5FN?usp=drive_link) |


## Citation

```
@article{vray2023source,
  title={Source-Free Open-Set Domain Adaptation for Histopathological Images via Distilling Self-Supervised Vision Transformer},
  author={Vray, Guillaume and Tomar, Devavrat and Bozorgtabar, Behzad and Thiran, Jean-Philippe},
  journal={arXiv preprint arXiv:2307.04596},
  year={2023}
}
```

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.

