# ----------------------
# PROJECT ROOT DIR
# ----------------------
project_root_dir = '.'

# ----------------------
# EXPERIMENT SAVE PATHS
# ----------------------
exp_root = 'log/'        # directory to store experiment output (checkpoints, logs, etc)
save_dir = 'log/'    # Evaluation save dir

# evaluation model path (for openset_test.py and openset_test_fine_grained.py, {} reserved for different options)
root_model_path = 'log/{}/checkpoints/{}_{}_{}.pth'

# -----------------------
# DATASET ROOT DIRS
# -----------------------
cifar_10_root = 'data/cifar10'                                          # CIFAR10
cifar_100_root = 'data/cifar100'                                        # CIFAR100
cub_root = 'data/CUB'                                                   # CUB
mnist_root = 'data/mnist/'                                              # MNIST
svhn_root = 'data/svhn'                                                 # SVHN
tin_train_root_dir = 'data/tinyimagenet/tiny-imagenet-200/train'        # TinyImageNet Train
tin_val_root_dir = 'data/tinyimagenet/tiny-imagenet-200/val/images'     # TinyImageNet Val

# ----------------------
# FGVC / IMAGENET OSR SPLITS
# ----------------------
osr_split_dir = 'data/open_set_splits'

# ----------------------
# PRETRAINED RESNET50 MODEL PATHS (For FGVC experiments)
# Weights can be downloaded from https://github.com/nanxuanzhao/Good_transfer
# ----------------------
imagenet_moco_path = 'models/moco_v2_800ep_pretrain.pth.tar'
places_moco_path = 'models/moco_v2_places.pth'
places_supervised_path = 'models/supervised_places.pth'
imagenet_supervised_path = 'models/supervised_imagenet.pth'