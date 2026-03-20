from torchvision import transforms
from utils import *
from datetime import datetime


class setting_config:
    """
    Configuration for DSNet-Mamba on ISIC skin lesion segmentation datasets.
    """

    network = 'dsnet_mamba'
    model_config = {
        'num_classes': 1,
        'input_channels': 3,
        'model_name': 's128',
        # DSNet backbone pretrained weights (ImageNet)
        'dsnet_pretrained_path': './pre_trained_weights/dsnet_pretrained.pth',
        # VMamba-Small pretrained weights
        'vmunet_pretrained_path': './pre_trained_weights/vmamba_small_e238_ema.pth',
    }

    datasets = 'isic17'   # options: 'isic17', 'isic18'
    if datasets == 'isic18':
        data_path = './data/isic2018/'
    elif datasets == 'isic17':
        data_path = './data/isic2017/'
    else:
        raise Exception('datasets is not right!')

    criterion = BceDiceLoss(wb=1, wd=1)

    num_classes = 1
    input_size_h = 256
    input_size_w = 256
    input_channels = 3
    distributed = False
    local_rank = -1
    num_workers = 4
    seed = 42
    world_size = None
    rank = None
    amp = False
    gpu_id = '0'
    batch_size = 32
    epochs = 300
    threshold = 0.5

    work_dir = 'results/' + network + '_' + datasets + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'

    print_interval = 20
    val_interval = 10
    save_interval = 100

    train_transformer = transforms.Compose([
        myNormalize(datasets, train=True),
        myToTensor(),
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[0, 360]),
        myResize(input_size_h, input_size_w)
    ])
    test_transformer = transforms.Compose([
        myNormalize(datasets, train=False),
        myToTensor(),
        myResize(input_size_h, input_size_w)
    ])

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
        T_max = 300
        eta_min = 1e-5
        last_epoch = -1