from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import cv2
import datetime
from collections import deque
import argparse
import math
import os
import logging
import time

from dcvt.datasets.detection import get_datasets_from_config
from dcvt.utils.evaluation import calculate_map
from dcvt.utils.losses import YoloBboxLoss
from dcvt.utils.common import read_config, set_determenistic, object_from_dict, write_temp_yml
from dcvt.utils.torch import initialize_weights, get_lr
from dcvt.infer import InferDetection
from dcvt.detection.evaluate import evaluate
from dcvt.models import get_model_type

from transform import get_transform


def train(config, et_logger=None):
    set_determenistic(42)
    
    train_config = config.train
    model_config = config.model

    train_preprocess, val_preprocess = get_transform(
        config.model.infer_sz_hw, 
        config=config
    )

    train_datasets, val_datasets, train_labels = get_datasets_from_config(
        config,
        train_preprocess=train_preprocess,
        val_preprocess=val_preprocess
    )
    config.model.labels = train_labels

    epochs = train_config.epochs
    # log_step=100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device {device}')
    logger.info(f'Using model config {config.model}')

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    subbatch_sz = train_config.batch // train_config.subdivisions

    train_loader = DataLoader(
        train_dataset, 
        batch_size=subbatch_sz, 
        shuffle=True, num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory, drop_last=False, 
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=subbatch_sz, 
        shuffle=False, num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory, drop_last=False, 
    )

    infer = InferDetection(
        model_config=model_config,
        device=device,
        nms_threshold=0.5, 
        conf_threshold=0.001 
    )
    
    if len(model_config.labels) == 1:
        chk_classes_names = f'cls_{model_config.labels[0]}'
    else:
        chk_classes_names = f'cls_{len(model_config.labels)}'

    if model_config.tiled:
        records_suffix = f'|{model_config.name}Tiled_{model_config.infer_sz_hw[1]}x{model_config.infer_sz_hw[0]}_{chk_classes_names}'
    else:
        records_suffix = f'|{model_config.name}_{model_config.infer_sz_hw[1]}x{model_config.infer_sz_hw[0]}_{chk_classes_names}'

    model_type = get_model_type(model_config.name)
    train_model = model_type(config=model_config, inference=False)
    initialize_weights(train_model)
    
    if 'pretrained' in config.train:
        pretrained_path = config.train['pretrained']
        loaded_data = torch.load(config.train['pretrained'])
        train_model.load_state_dict(loaded_data['model_state'])
        logger.info(f'Loaded weights from {pretrained_path}')

    train_model.to(device)

    optimizer = object_from_dict(
        d=train_config.optimizer,
        params=train_model.parameters(),
        lr=float(train_config.lr)
    )

    scheduler = object_from_dict(
        d=train_config.scheduler,
        optimizer=optimizer, 
        eta_min=1e-8
    )

    # Loss config
    ## NOTE - this one must return tuple:
    ##          - main loss with gradient
    ##          - dict of loss parts or None
    criterion = YoloBboxLoss(
        device=device,
        batch=subbatch_sz, 
        config=config
    )
    
    # Checkpoints config
    saved_models = deque()
    os.makedirs(train_config.checkpoints_dir, exist_ok=True)

    global_step = 0    
    best_model_name = None
    best_model_value = 0
    
    ARTIFACTS_CACHE_DIR = train_config.artifacts_dir
    os.makedirs(ARTIFACTS_CACHE_DIR, exist_ok=True)
    
    ## Log config file
    et_logger.log_artifact(
        local_path=write_temp_yml(config.to_dict())
    )

    try:
        if et_logger is not None:
            params = {
                'optimizer': train_config.optimizer.type,
                'scheduler': train_config.scheduler.type,
                'initial_lr': train_config.lr,
                'epochs': train_config.epochs,
                'in_width': model_config.infer_sz_hw[1],
                'in_height': model_config.infer_sz_hw[0],
            }

            et_logger.log_hyperparams(params)
            
        for epoch in range(epochs):
            epoch_loss = []

            _losses = {}
            curr_lr = get_lr(optimizer)
            print(f'Current lr: {curr_lr}')

            train_losses = []
            val_losses = []

            ### TRAIN PART
            train_model.train()
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', ncols=80) as pbar:
                for batch in train_loader:
                    global_step += 1
                    images = batch[0].to(device=device)
                    bboxes = batch[1].to(device=device)
                    c_batch_sz = images.shape[0]
                    # _batch_rate = (c_batch_sz * train_config.subdivisions)
                    _batch_rate = train_config.subdivisions

                    bboxes_pred = train_model(images)
                    main_loss, loss_parts = criterion(bboxes_pred, bboxes)
                    main_loss = main_loss / _batch_rate
                    main_loss.backward()
                    
                    train_losses.append(main_loss.item())
                    
                    pbar.update(c_batch_sz)

                    # Criterion provides named parts
                    if loss_parts is not None:
                        for loss_name, loss_val in loss_parts.items():
                            _name = f'train/loss_{loss_name}'
                            if _name not in _losses:
                                _losses[_name] = []
                            
                            _losses[_name].append(loss_val.item()/_batch_rate)

                    if global_step % train_config.subdivisions == 0:
                        optimizer.step()
                        optimizer.zero_grad()

            ### VALIDATION PART
            train_model.eval()
            with torch.no_grad():
                with tqdm(total=n_val, desc=f'Val Epoch {epoch + 1}/{epochs}', unit='img', ncols=70) as pbar:
                    for batch in val_loader:
                        images = batch[0].to(device=device)
                        bboxes = batch[1].to(device=device)
                        c_batch_sz = images.shape[0]
                        # _batch_rate = (c_batch_sz * train_config.subdivisions)
                        _batch_rate = train_config.subdivisions
                        
                        bboxes_pred = train_model(images)
                        main_loss, loss_parts = criterion(bboxes_pred, bboxes)
                        main_loss = main_loss / _batch_rate

                        val_losses.append(main_loss.item())
                        pbar.update(c_batch_sz)


                        # Criterion provides named parts
                        if loss_parts is not None:
                            for loss_name, loss_val in loss_parts.items():
                                _name = f'val/loss_{loss_name}'
                                if _name not in _losses:
                                    _losses[_name] = []
                                
                                _losses[_name].append(loss_val.item()/_batch_rate)

            # Average losses

            _losses = {k: np.mean(v) for k, v in _losses.items()}

            epoch_loss = np.mean(train_losses)
            val_epoch_loss = np.mean(val_losses)

            logger.info(f'Train epoch loss: {epoch_loss}')
            logger.info(f'Valid epoch loss: {val_epoch_loss}')

            map50 = 0
            map75 = 0
            if epoch >= train_config.get('skip_eval', 0):
                # eval_model.load_state_dict(train_model.state_dict())
                infer.update_model_state(train_model.state_dict())

                metrics = evaluate(
                    infer=infer,
                    eval_datasets=val_datasets
                )
                
                map50 = metrics.map50.value
                map75 = metrics.map75.value
                
            if et_logger is not None:
                send_metrics = {
                    'lr': curr_lr,
                    'train/loss': epoch_loss,
                    'val/loss': val_epoch_loss,
                    'eval/ap50': map50,
                    'eval/ap75': map75,
                }
                
                send_metrics.update(_losses)
                
                et_logger.log_metrics(send_metrics, epoch)

            scheduler.step()
            # scheduler.step(val_epoch_loss)

            ap_value = round(map50, 3)
            checkpoint_suffix = train_config.get('checkpoints_suffix', '')
            if model_config.tiled:
                chk_name = f'{model_config.name}Tiled_epoch_{epoch + 1}_ap_{ap_value}_{checkpoint_suffix}.pth'
            else:
                chk_name = f'{model_config.name}_epoch_{epoch + 1}_ap_{ap_value}_{checkpoint_suffix}.pth'

            # Checkpoint save
            save_path = os.path.join(
                train_config.checkpoints_dir, 
                chk_name
            )
            save_data = {
                'model_state': train_model.state_dict(),
                'model_config': dict(model_config)
            }

            torch.save(
                save_data, 
                save_path,
                _use_new_zipfile_serialization=False    # For backward compatibility
            )
            logger.info(f'Checkpoint {epoch + 1} saved !')
            
            ## LOG ARTIFACTS
            best_map_model_name = f'{model_config.name}_best_mAP.pth'
            best_map_model_path = os.path.join(
                ARTIFACTS_CACHE_DIR, 
                best_map_model_name
            )
            torch.save(
                save_data, 
                best_map_model_path,
                _use_new_zipfile_serialization=False
            )                
            et_logger.log_artifact(
                local_path=best_map_model_path
            )
                
            # Best model checkpoints
            if best_model_value < (map50 * 2 + map75):
                best_model_value = map50 * 2 + map75

                if best_model_name is not None:
                    try:
                        os.remove(best_model_name)
                    except:
                        pass
                
                best_model_name = f'{model_config.name}_best_ep{epoch + 1}_ap_{ap_value}_{checkpoint_suffix}.pth'
                best_model_name = os.path.join(
                    train_config.checkpoints_dir, 
                    best_model_name
                )
                torch.save(
                    save_data, 
                    best_model_name,
                    _use_new_zipfile_serialization=False
                )
            
            # Cleaning
            saved_models.append(save_path)
            if len(saved_models) > train_config.keep_checkpoint_max > 0:
                model_to_remove = saved_models.popleft()
                try:
                    os.remove(model_to_remove)
                except:
                    logger.info(f'failed to remove {model_to_remove}')
    except Exception as e:
        print(f'Failed to train: {e}')
        raise


def get_args():
    parser = argparse.ArgumentParser(description='Train the Model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str)
    args = vars(parser.parse_args())
    return args


def init_logger(log_file=None, log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    logging.basicConfig(
        level=log_level,
        format=fmt,
        # filename=log_file,
        # filemode=mode
    )

    logger = logging.getLogger(__name__)
    return logger


if __name__ == "__main__":
    logger = init_logger()
    args = get_args()
    config = read_config(args['config'])

    exp_logger = object_from_dict(
        d=config.metrics_logger
        
    )

    try:
        train(
            config=config, et_logger=exp_logger
        )
    except Exception as e:
        print(f'Train failed: {e}')
        exp_logger.finalize(status='FAILED')
        raise
    
    exp_logger.finalize(status='FINISHED')
    