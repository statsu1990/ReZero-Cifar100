import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import albumentations as alb
from albumentations.augmentations import transforms as albtr
from albumentations.pytorch import ToTensor as albToTensor

import torch.nn as nn
import torch.optim as optim

from data import cifar, torch_data_utils
from model import preact_resnet, rezero_preact_resnet, rezero2_preact_resnet
from train import training


def get_checkpoint(path):
    cp = torch.load(path, map_location=lambda storage, loc: storage)
    return cp

def make_PreactResnet():
    DOWNLOAD = False
    
    CHECKPOINT_PATH = None #'checkpoint', None
    FINE_TURNING = False
    CP = get_checkpoint(CHECKPOINT_PATH) if CHECKPOINT_PATH is not None else None

    ## data
    # transformer
    tr_transformer = alb.Compose([
                            albtr.Flip(p=0.5),
                            albtr.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                            albtr.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
                            albToTensor()
                            ])
    ts_transformer = alb.Compose([
                            albtr.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
                            albToTensor()
                            ])
    # dataset
    tr_ds = cifar.get_dataset_cifar100(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer)
    ts_ds = cifar.get_dataset_cifar100(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer)

    ## model
    model = preact_resnet.PreActResNet18(num_classes=100)
    if CP is not None:
        model.load_state_dict(CP['state_dict'])
    USE_LABEL = False

    ## training
    TR_BATCH_SIZE = 128
    TS_BATCH_SIZE = 512
    tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
    ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

    LR = 0.1
    opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    if CP is not None:
        if not FINE_TURNING:
            opt.load_state_dict(CP['optimizer'])
    tr_criterion = nn.CrossEntropyLoss()
    vl_criterion = nn.CrossEntropyLoss()

    grad_accum_steps = 1
    start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
    EPOCHS = 200

    warmup_epoch=1
    step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[60, 120, 160], gamma=0.2) #learning rate decay

    filename_head = 'preact18_'

    model = training.train_model(model, tr_loader, ts_loader, USE_LABEL,
                                 opt, tr_criterion, vl_criterion, 
                                 grad_accum_steps, start_epoch, EPOCHS, 
                                 warmup_epoch, step_scheduler, filename_head)
            
    # save
    torch.save(model.state_dict(), filename_head + '_model')

    return

def make_RezeroPreactResnet():
    DOWNLOAD = False
    
    CHECKPOINT_PATH = None #'checkpoint', None
    FINE_TURNING = False
    CP = get_checkpoint(CHECKPOINT_PATH) if CHECKPOINT_PATH is not None else None

    ## data
    # transformer
    tr_transformer = alb.Compose([
                            albtr.Flip(p=0.5),
                            albtr.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                            albtr.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
                            albToTensor()
                            ])
    ts_transformer = alb.Compose([
                            albtr.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
                            albToTensor()
                            ])
    # dataset
    tr_ds = cifar.get_dataset_cifar100(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer)
    ts_ds = cifar.get_dataset_cifar100(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer)

    ## model
    model = rezero_preact_resnet.PreActResNet18(num_classes=100)
    if CP is not None:
        model.load_state_dict(CP['state_dict'])
    model = model.cuda()
    USE_LABEL = False

    ## training
    TR_BATCH_SIZE = 128
    TS_BATCH_SIZE = 512
    tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
    ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

    LR = 0.1
    opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    if CP is not None:
        if not FINE_TURNING:
            opt.load_state_dict(CP['optimizer'])
    tr_criterion = nn.CrossEntropyLoss()
    vl_criterion = nn.CrossEntropyLoss()

    grad_accum_steps = 1
    start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
    EPOCHS = 200

    warmup_epoch=1
    step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[60, 120, 160], gamma=0.2) #learning rate decay

    filename_head = 'rezero_preact18_'

    model = training.train_model(model, tr_loader, ts_loader, USE_LABEL,
                                 opt, tr_criterion, vl_criterion, 
                                 grad_accum_steps, start_epoch, EPOCHS, 
                                 warmup_epoch, step_scheduler, filename_head)
            
    # save
    torch.save(model.state_dict(), filename_head + '_model')

    return

def make_Rezero2PreactResnet():
    DOWNLOAD = False
    
    CHECKPOINT_PATH = None #'checkpoint', None
    FINE_TURNING = False
    CP = get_checkpoint(CHECKPOINT_PATH) if CHECKPOINT_PATH is not None else None

    ## data
    # transformer
    tr_transformer = alb.Compose([
                            albtr.Flip(p=0.5),
                            albtr.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                            albtr.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
                            albToTensor()
                            ])
    ts_transformer = alb.Compose([
                            albtr.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
                            albToTensor()
                            ])
    # dataset
    tr_ds = cifar.get_dataset_cifar100(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer)
    ts_ds = cifar.get_dataset_cifar100(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer)

    ## model
    model = rezero2_preact_resnet.PreActResNet18(num_classes=100)
    if CP is not None:
        model.load_state_dict(CP['state_dict'])
    model = model.cuda()
    USE_LABEL = False

    ## training
    TR_BATCH_SIZE = 128
    TS_BATCH_SIZE = 512
    tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
    ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

    LR = 0.1
    opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    if CP is not None:
        if not FINE_TURNING:
            opt.load_state_dict(CP['optimizer'])
    tr_criterion = nn.CrossEntropyLoss()
    vl_criterion = nn.CrossEntropyLoss()

    grad_accum_steps = 1
    start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
    EPOCHS = 200

    warmup_epoch=1
    step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[60, 120, 160], gamma=0.2) #learning rate decay

    filename_head = 'rezero2_preact18_'

    model = training.train_model(model, tr_loader, ts_loader, USE_LABEL,
                                 opt, tr_criterion, vl_criterion, 
                                 grad_accum_steps, start_epoch, EPOCHS, 
                                 warmup_epoch, step_scheduler, filename_head)
            
    # save
    torch.save(model.state_dict(), filename_head + '_model')

    return

def make_PreactResnet50():
    DOWNLOAD = False
    
    CHECKPOINT_PATH = None #'checkpoint', None
    FINE_TURNING = False
    CP = get_checkpoint(CHECKPOINT_PATH) if CHECKPOINT_PATH is not None else None

    ## data
    # transformer
    tr_transformer = alb.Compose([
                            albtr.Flip(p=0.5),
                            albtr.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                            albtr.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
                            albToTensor()
                            ])
    ts_transformer = alb.Compose([
                            albtr.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
                            albToTensor()
                            ])
    # dataset
    tr_ds = cifar.get_dataset_cifar100(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer)
    ts_ds = cifar.get_dataset_cifar100(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer)

    ## model
    model = preact_resnet.PreActResNet50(num_classes=100)
    if CP is not None:
        model.load_state_dict(CP['state_dict'])
    USE_LABEL = False

    ## training
    TR_BATCH_SIZE = 128
    TS_BATCH_SIZE = 512
    tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
    ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

    LR = 0.1
    opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    if CP is not None:
        if not FINE_TURNING:
            opt.load_state_dict(CP['optimizer'])
    tr_criterion = nn.CrossEntropyLoss()
    vl_criterion = nn.CrossEntropyLoss()

    grad_accum_steps = 1
    start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
    EPOCHS = 200

    warmup_epoch=1
    step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[60, 120, 160], gamma=0.2) #learning rate decay

    filename_head = 'preact50_'

    model = training.train_model(model, tr_loader, ts_loader, USE_LABEL,
                                 opt, tr_criterion, vl_criterion, 
                                 grad_accum_steps, start_epoch, EPOCHS, 
                                 warmup_epoch, step_scheduler, filename_head)
            
    # save
    torch.save(model.state_dict(), filename_head + '_model')

    return

def make_RezeroPreactResnet50():
    DOWNLOAD = False
    
    CHECKPOINT_PATH = None #'checkpoint', None
    FINE_TURNING = False
    CP = get_checkpoint(CHECKPOINT_PATH) if CHECKPOINT_PATH is not None else None

    ## data
    # transformer
    tr_transformer = alb.Compose([
                            albtr.Flip(p=0.5),
                            albtr.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                            albtr.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
                            albToTensor()
                            ])
    ts_transformer = alb.Compose([
                            albtr.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
                            albToTensor()
                            ])
    # dataset
    tr_ds = cifar.get_dataset_cifar100(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer)
    ts_ds = cifar.get_dataset_cifar100(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer)

    ## model
    model = rezero_preact_resnet.PreActResNet50(num_classes=100)
    if CP is not None:
        model.load_state_dict(CP['state_dict'])
    USE_LABEL = False

    ## training
    TR_BATCH_SIZE = 128
    TS_BATCH_SIZE = 512
    tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
    ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

    LR = 0.1
    opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    if CP is not None:
        if not FINE_TURNING:
            opt.load_state_dict(CP['optimizer'])
    tr_criterion = nn.CrossEntropyLoss()
    vl_criterion = nn.CrossEntropyLoss()

    grad_accum_steps = 1
    start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
    EPOCHS = 200

    warmup_epoch=1
    step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[60, 120, 160], gamma=0.2) #learning rate decay

    filename_head = 'rezero_preact50_'

    model = training.train_model(model, tr_loader, ts_loader, USE_LABEL,
                                 opt, tr_criterion, vl_criterion, 
                                 grad_accum_steps, start_epoch, EPOCHS, 
                                 warmup_epoch, step_scheduler, filename_head)
            
    # save
    torch.save(model.state_dict(), filename_head + '_model')

    return

def make_PreactResnet152():
    DOWNLOAD = False
    
    CHECKPOINT_PATH = None #'checkpoint', None
    FINE_TURNING = False
    CP = get_checkpoint(CHECKPOINT_PATH) if CHECKPOINT_PATH is not None else None

    ## data
    # transformer
    tr_transformer = alb.Compose([
                            albtr.Flip(p=0.5),
                            albtr.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                            albtr.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
                            albToTensor()
                            ])
    ts_transformer = alb.Compose([
                            albtr.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
                            albToTensor()
                            ])
    # dataset
    tr_ds = cifar.get_dataset_cifar100(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer)
    ts_ds = cifar.get_dataset_cifar100(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer)

    ## model
    model = preact_resnet.PreActResNet152(num_classes=100)
    if CP is not None:
        model.load_state_dict(CP['state_dict'])
    USE_LABEL = False

    ## training
    TR_BATCH_SIZE = 128
    TS_BATCH_SIZE = 512
    tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
    ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

    LR = 0.1
    opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    if CP is not None:
        if not FINE_TURNING:
            opt.load_state_dict(CP['optimizer'])
    tr_criterion = nn.CrossEntropyLoss()
    vl_criterion = nn.CrossEntropyLoss()

    grad_accum_steps = 1
    start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
    EPOCHS = 200

    warmup_epoch=1
    step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[60, 120, 160], gamma=0.2) #learning rate decay

    filename_head = 'preact152_'

    model = training.train_model(model, tr_loader, ts_loader, USE_LABEL,
                                 opt, tr_criterion, vl_criterion, 
                                 grad_accum_steps, start_epoch, EPOCHS, 
                                 warmup_epoch, step_scheduler, filename_head)
            
    # save
    torch.save(model.state_dict(), filename_head + '_model')

    return

def make_RezeroPreactResnet152():
    DOWNLOAD = False
    
    CHECKPOINT_PATH = None #'checkpoint', None
    FINE_TURNING = False
    CP = get_checkpoint(CHECKPOINT_PATH) if CHECKPOINT_PATH is not None else None

    ## data
    # transformer
    tr_transformer = alb.Compose([
                            albtr.Flip(p=0.5),
                            albtr.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                            albtr.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
                            albToTensor()
                            ])
    ts_transformer = alb.Compose([
                            albtr.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
                            albToTensor()
                            ])
    # dataset
    tr_ds = cifar.get_dataset_cifar100(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer)
    ts_ds = cifar.get_dataset_cifar100(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer)

    ## model
    model = rezero_preact_resnet.PreActResNet152(num_classes=100)
    if CP is not None:
        model.load_state_dict(CP['state_dict'])
    USE_LABEL = False

    ## training
    TR_BATCH_SIZE = 128
    TS_BATCH_SIZE = 512
    tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
    ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

    LR = 0.1
    opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    if CP is not None:
        if not FINE_TURNING:
            opt.load_state_dict(CP['optimizer'])
    tr_criterion = nn.CrossEntropyLoss()
    vl_criterion = nn.CrossEntropyLoss()

    grad_accum_steps = 1
    start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
    EPOCHS = 200

    warmup_epoch=1
    step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[60, 120, 160], gamma=0.2) #learning rate decay

    filename_head = 'rezero_preact152_'

    model = training.train_model(model, tr_loader, ts_loader, USE_LABEL,
                                 opt, tr_criterion, vl_criterion, 
                                 grad_accum_steps, start_epoch, EPOCHS, 
                                 warmup_epoch, step_scheduler, filename_head)
            
    # save
    torch.save(model.state_dict(), filename_head + '_model')

    return

def make_Rezero2PreactResnet152():
    DOWNLOAD = False
    
    CHECKPOINT_PATH = None #'checkpoint', None
    FINE_TURNING = False
    CP = get_checkpoint(CHECKPOINT_PATH) if CHECKPOINT_PATH is not None else None

    ## data
    # transformer
    tr_transformer = alb.Compose([
                            albtr.Flip(p=0.5),
                            albtr.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                            albtr.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
                            albToTensor()
                            ])
    ts_transformer = alb.Compose([
                            albtr.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
                            albToTensor()
                            ])
    # dataset
    tr_ds = cifar.get_dataset_cifar100(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer)
    ts_ds = cifar.get_dataset_cifar100(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer)

    ## model
    model = rezero2_preact_resnet.PreActResNet152(num_classes=100)
    if CP is not None:
        model.load_state_dict(CP['state_dict'])
    USE_LABEL = False

    ## training
    TR_BATCH_SIZE = 128
    TS_BATCH_SIZE = 512
    tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
    ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

    LR = 0.1
    opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    if CP is not None:
        if not FINE_TURNING:
            opt.load_state_dict(CP['optimizer'])
    tr_criterion = nn.CrossEntropyLoss()
    vl_criterion = nn.CrossEntropyLoss()

    grad_accum_steps = 1
    start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
    EPOCHS = 200

    warmup_epoch=1
    step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[60, 120, 160], gamma=0.2) #learning rate decay

    filename_head = 'rezero2_preact152_'

    model = training.train_model(model, tr_loader, ts_loader, USE_LABEL,
                                 opt, tr_criterion, vl_criterion, 
                                 grad_accum_steps, start_epoch, EPOCHS, 
                                 warmup_epoch, step_scheduler, filename_head)
            
    # save
    torch.save(model.state_dict(), filename_head + '_model')

    return
