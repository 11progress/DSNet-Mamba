from torchvision import transforms
from utils import *
from datasets.dataset import Synapse_dataset, RandomGenerator
import os
from datetime import datetime

SYNAPSE_CLASS_NAMES = {
    1: 'Aorta',
    2: 'Gallbladder',
    3: 'Kidney(L)',
    4: 'Kidney(R)',
    5: 'Liver',
    6: 'Pancreas',
    7: 'Spleen',
    8: 'Stomach'
}


class setting_config:
    """
    Configuration for DSNet-Mamba on the Synapse multi-organ segmentation dataset.
    """

    network = 'dsnet_mamba'
    model_config = {
        'num_classes': 9,
        'input_channels': 1,
        'model_name': 's128',
        # DSNet backbone pretrained weights (ImageNet)
        'dsnet_pretrained_path': './pre_trained_weights/dsnet_pretrained.pth',
        # VMamba-Small pretrained weights
        'vmunet_pretrained_path': './pre_trained_weights/vmamba_small_e238_ema.pth',
    }

    datasets_name = 'synapse'
    data_path = './data/Synapse/train_npz/'
    list_dir = './data/Synapse/lists/lists_Synapse/'
    volume_path = './data/Synapse/test_vol_h5/'
    datasets = Synapse_dataset

    criterion = CeDiceLoss(num_classes=9, loss_weight=[1.0, 1.0])

    num_classes = 9
    input_size_h = 224
    input_size_w = 224
    input_channels = 1
    distributed = False
    local_rank = -1
    num_workers = 4
    seed = 42
    world_size = None
    rank = None
    amp = False
    gpu_id = '0'
    batch_size = 32
    epochs = 200
    z_spacing = 1
    threshold = 0.5

    work_dir = 'results/' + network + '_' + datasets_name + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'

    print_interval = 20
    val_interval = 10
    save_interval = 100

    opt = 'AdamW'
    assert opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], \
        'Unsupported optimizer!'
    if opt == 'AdamW':
        lr = 0.001
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 1e-2
        amsgrad = False

    sch = 'CosineAnnealingLR'
    if sch == 'CosineAnnealingLR':
        T_max = 200
        eta_min = 1e-5
        last_epoch = -1