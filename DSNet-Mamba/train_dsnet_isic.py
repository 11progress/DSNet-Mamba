import torch
from torch.utils.data import DataLoader
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
from models.dsnet_medical import DSNetMedical

from engine_isic import *
import os
import sys

from utils import *
from configs.config_setting_isic import setting_config

import warnings
warnings.filterwarnings("ignore")


def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    for d in [checkpoint_dir, outputs]:
        os.makedirs(d, exist_ok=True)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.num_workers
    )
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers,
        drop_last=False
    )

    print('#----------Preparing Model----------#')
    model_cfg = config.model_config
    model = DSNetMedical(
        num_classes=model_cfg['num_classes'],
        model_name=model_cfg.get('model_name', 's128'),
        dsnet_pretrained_path=model_cfg.get('dsnet_pretrained_path', None),
        vmunet_pretrained_path=model_cfg.get('vmunet_pretrained_path', None),
    ).cuda()

    cal_params_flops(model, 256, logger)

    print('#----------Preparing loss, opt, sch----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    best_dsc   = 0.0
    start_epoch = 1
    best_epoch  = 1

    # Resume training if checkpoint exists
    if os.path.exists(resume_model):
        print('#----------Resuming from checkpoint----------#')
        checkpoint = torch.load(resume_model, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch = saved_epoch + 1
        best_dsc   = checkpoint['best_dsc']
        best_epoch = checkpoint['best_epoch']
        logger.info(
            f"Resumed from {resume_model}. "
            f"epoch: {saved_epoch}, best_dsc: {best_dsc:.4f}, best_epoch: {best_epoch}"
        )

    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):
        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader, model, criterion, optimizer,
            scheduler, epoch, step, logger, config, writer
        )

        dsc = val_one_epoch(
            val_loader, model, criterion, epoch, logger, config
        )

        if dsc > best_dsc:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            best_dsc   = dsc
            best_epoch = epoch
            logger.info(f'[BEST] epoch={best_epoch}, best_dsc={best_dsc:.4f}')

        torch.save(
            {
                'epoch': epoch,
                'best_dsc': best_dsc,
                'best_epoch': best_epoch,
                'loss': dsc,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            },
            os.path.join(checkpoint_dir, 'latest.pth')
        )

    # Testing with best checkpoint
    best_ckpt = os.path.join(checkpoint_dir, 'best.pth')
    if os.path.exists(best_ckpt):
        print('#----------Testing----------#')
        model.load_state_dict(torch.load(best_ckpt, map_location='cpu'))
        test_results = test_one_epoch(
            val_loader, model, criterion, logger, config
        )

        total_params = sum(p.numel() for p in model.parameters())
        params_M = total_params / 1e6

        from thop import profile
        dummy = torch.randn(1, 3, 256, 256).cuda()
        flops, _ = profile(model, inputs=(dummy,))
        flops_G = flops / 1e9

        print('\n' + '=' * 80)
        print('Final Test Results:')
        print('=' * 80)
        print(f"{'mIoU(%)':<10} {'DSC(%)':<10} {'Acc(%)':<10} {'Spe(%)':<10} {'Sen(%)':<10} {'Params(M)':<12} {'FLOPs(G)':<10}")
        print('-' * 80)
        print(
            f"{test_results['miou']*100:<10.2f} {test_results['dsc']*100:<10.2f} "
            f"{test_results['accuracy']*100:<10.2f} {test_results['specificity']*100:<10.2f} "
            f"{test_results['sensitivity']*100:<10.2f} {params_M:<12.2f} {flops_G:<10.2f}"
        )
        print('=' * 80)

        logger.info(
            f"Final Test Results - "
            f"mIoU: {test_results['miou']*100:.2f}%, DSC: {test_results['dsc']*100:.2f}%, "
            f"Acc: {test_results['accuracy']*100:.2f}%, Spe: {test_results['specificity']*100:.2f}%, "
            f"Sen: {test_results['sensitivity']*100:.2f}%, "
            f"Params: {params_M:.2f}M, FLOPs: {flops_G:.2f}G"
        )

        os.rename(
            best_ckpt,
            os.path.join(checkpoint_dir, f'best-epoch{best_epoch}-dsc{best_dsc:.4f}.pth')
        )


if __name__ == '__main__':
    config = setting_config
    main(config)