import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.dataset import Synapse_dataset, RandomGenerator
from tensorboardX import SummaryWriter
from models.dsnet_medical import DSNetMedical

from engine_synapse import *
import os
import sys
import glob

from utils import *
from configs.config_setting_synapse import setting_config, SYNAPSE_CLASS_NAMES

import warnings
warnings.filterwarnings("ignore")


def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    test_save_path = os.path.join(outputs, 'test_results')
    for d in [checkpoint_dir, outputs, test_save_path]:
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
    train_dataset = config.datasets(
        base_dir=config.data_path,
        list_dir=config.list_dir,
        split="train",
        transform=transforms.Compose(
            [RandomGenerator(output_size=[config.input_size_h, config.input_size_w])]
        )
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.num_workers
    )

    val_dataset = config.datasets(
        base_dir=config.volume_path,
        split="test_vol",
        list_dir=config.list_dir
    )
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

    cal_params_flops(model, config.input_size_h, logger)

    print('#----------Preparing loss, opt, sch----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    start_epoch = 1
    best_dice  = 0.0
    best_epoch = 0

    # Resume training if checkpoint exists
    if os.path.exists(resume_model):
        print('#----------Resuming from checkpoint----------#')
        checkpoint = torch.load(resume_model, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = int(checkpoint.get('epoch', 0))
        start_epoch = saved_epoch + 1
        best_dice  = float(checkpoint.get('best_dice', 0.0))
        best_epoch = int(checkpoint.get('best_epoch', 0))
        last_loss  = float(checkpoint.get('loss', -1.0))
        logger.info(
            f"Resumed from {resume_model}. "
            f"epoch: {saved_epoch}, loss: {last_loss:.4f}, "
            f"best_dice: {best_dice:.4f}, best_epoch: {best_epoch}"
        )

    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):
        torch.cuda.empty_cache()

        loss = train_one_epoch(
            train_loader, model, criterion, optimizer,
            scheduler, epoch, logger, config, scaler=None
        )

        if epoch % config.val_interval == 0:
            mean_dice, mean_hd95 = val_one_epoch(
                val_dataset, val_loader, model, epoch,
                logger, config, test_save_path=test_save_path, val_or_test=True
            )
            if mean_dice > best_dice:
                best_dice  = float(mean_dice)
                best_epoch = int(epoch)
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_dice.pth'))
                logger.info(
                    f"[BEST] epoch={best_epoch}, best_dice={best_dice:.6f}, "
                    f"best_hd95={float(mean_hd95):.6f}"
                )

        torch.save(
            {
                'epoch': epoch,
                'loss': loss,
                'best_dice': best_dice,
                'best_epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            },
            os.path.join(checkpoint_dir, 'latest.pth')
        )

    # Testing with best checkpoint
    best_ckpt = os.path.join(checkpoint_dir, 'best_dice.pth')
    if not os.path.exists(best_ckpt):
        candidates = glob.glob(os.path.join(checkpoint_dir, 'best-dice-epoch*.pth'))
        if candidates:
            candidates.sort(key=os.path.getmtime, reverse=True)
            best_ckpt = candidates[0]

    if os.path.exists(best_ckpt):
        print('#----------Testing (best_dice)----------#')
        logger.info(f"Testing with best checkpoint: {best_ckpt} (best_epoch={best_epoch}, best_dice={best_dice:.6f})")
        model.load_state_dict(torch.load(best_ckpt, map_location='cpu'))

        test_results = test_one_epoch_synapse(
            val_dataset, val_loader, model, logger, config,
            test_save_path=test_save_path
        )

        total_params = sum(p.numel() for p in model.parameters())
        params_M = total_params / 1e6

        from thop import profile
        dummy = torch.randn(1, model_cfg['input_channels'], config.input_size_h, config.input_size_w).cuda()
        flops, _ = profile(model, inputs=(dummy,))
        flops_G = flops / 1e9

        print('\n' + '=' * 100)
        print('Final Test Results for Synapse Dataset (best_dice checkpoint):')
        print('=' * 100)
        print(f"{'Class':<15} {'DSC(%)':<12} {'HD95(mm)':<12}")
        print('-' * 100)
        for i in range(1, config.num_classes):
            class_name = SYNAPSE_CLASS_NAMES.get(i, f'Class_{i}')
            dice = test_results['class_metrics'][i - 1][0] * 100
            hd95 = test_results['class_metrics'][i - 1][1]
            print(f"{class_name:<15} {dice:<12.2f} {hd95:<12.2f}")
        print('-' * 100)
        print(f"{'Mean':<15} {test_results['mean_dice'] * 100:<12.2f} {test_results['mean_hd95']:<12.2f}")
        print('=' * 100)
        print(f"\n{'Params(M)':<15} {params_M:<12.2f}")
        print(f"{'FLOPs(G)':<15} {flops_G:<12.2f}")
        print('=' * 100)

        log_str = "Final Test Results (best_dice) - "
        for i in range(1, config.num_classes):
            class_name = SYNAPSE_CLASS_NAMES.get(i, f'Class_{i}')
            dice = test_results['class_metrics'][i - 1][0] * 100
            hd95 = test_results['class_metrics'][i - 1][1]
            log_str += f"{class_name}: DSC={dice:.2f}%, HD95={hd95:.2f}mm; "
        log_str += f"Mean: DSC={test_results['mean_dice'] * 100:.2f}%, HD95={test_results['mean_hd95']:.2f}mm; "
        log_str += f"Params: {params_M:.2f}M, FLOPs: {flops_G:.2f}G"
        logger.info(log_str)

        final_name = (
            f'best-dice-epoch{best_epoch}'
            f'-dice{test_results["mean_dice"]:.4f}'
            f'-hd95{test_results["mean_hd95"]:.4f}.pth'
        )
        os.rename(best_ckpt, os.path.join(checkpoint_dir, final_name))
    else:
        logger.info("No best_dice.pth found. Skip testing.")


if __name__ == '__main__':
    config = setting_config
    main(config)