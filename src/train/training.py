import torch
import numpy as np

from .scheduler import WarmUpLR
from .train_utils import save_log, save_checkpoint
from tqdm import tqdm


def trainer(net, loader, criterion, optimizer, grad_accum_steps, warmup_scheduler, use_label=False):
    net.train()

    #if hasattr(net, 'apply'):
    #    net.apply()

    total_loss = 0
    num_correct = 0
    num_total = 0

    optimizer.zero_grad()
    for batch_idx, (imgs, labels) in enumerate(tqdm(loader)):
        if warmup_scheduler is not None:
            warmup_scheduler.step()

        imgs = imgs.cuda()
        labels = labels.cuda()

        if use_label:
            outputs = net(imgs, labels)[0]
        else:
            outputs = net(imgs)
        loss = criterion(outputs, labels)

        loss = loss / grad_accum_steps
        loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # loss
        total_loss += loss.item() * grad_accum_steps
        
        # score
        _, predicted = outputs.max(1)
        num_total += labels.size(0)
        num_correct += predicted.eq(labels).sum().item()

    # loss
    total_loss = total_loss / (batch_idx + 1)

    # score
    score = 100. * num_correct / num_total

    print('Train Loss: %.3f | Acc: %.3f%% (%d/%d)' % (total_loss, score, num_correct, num_total))
    
    return total_loss, score

def tester(net, loader, criterion, use_label=False, return_value=False):
    net.eval()
    total_loss = 0
    num_correct = 0
    num_total = 0

    true_label = []
    pred_label = []
    pred_logit = []

    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(tqdm(loader)):
            imgs = imgs.cuda()
            labels = labels.cuda()

            if use_label:
                outputs = net(imgs, labels)[0]
            else:
                outputs = net(imgs)
            loss = criterion(outputs, labels)

            # loss
            total_loss += loss.item()
        
            # score
            _, predicted = outputs.max(1)
            num_total += labels.size(0)
            num_correct += predicted.eq(labels).sum().item()

            if return_value:
                true_label.append(labels.cpu().numpy())
                pred_label.append(predicted.cpu().numpy())
                pred_logit.append(outputs.cpu().numpy())

    # loss
    total_loss = total_loss / (batch_idx + 1)
    # score
    score = 100. * num_correct / num_total
    print('Test Loss: %.3f | Acc: %.3f%% (%d/%d)' % (total_loss, score, num_correct, num_total))
    
    if return_value:
        true_label = np.concatenate(true_label)
        pred_label = np.concatenate(pred_label)
        pred_logit = np.concatenate(pred_logit)
        values = np.concatenate([true_label[:,None], pred_label[:,None], pred_logit], axis=1)
        return total_loss, score, values
    else:
        return total_loss, score

def train_model(net, tr_loader, vl_loader, use_label, 
                optimizer, tr_criterion, vl_criterion, 
                grad_accum_steps, start_epoch, epochs, 
                warmup_epoch, step_scheduler, filename_head=''):
    net = net.cuda()

    # warmup_scheduler
    if start_epoch < warmup_epoch:
        warmup_scheduler = WarmUpLR(optimizer, len(tr_loader) * warmup_epoch)
    else:
        warmup_scheduler = None
    
    # train
    loglist = []
    for epoch in range(start_epoch, epochs):
        if epoch > warmup_epoch - 1:
            warm_sch = None
            step_scheduler.step()
        else:
            warm_sch = warmup_scheduler

        print('epoch ', epoch)
        for param_group in optimizer.param_groups:
            print('lr ', param_group['lr'])
            now_lr = param_group['lr']

        tr_log = trainer(net, tr_loader, tr_criterion, optimizer, grad_accum_steps, warm_sch, use_label)
        vl_log = tester(net, vl_loader, vl_criterion, use_label=False)

        # save checkpoint
        save_checkpoint(epoch, net, optimizer, filename_head + 'checkpoint')

        # save log
        loglist.append([epoch] + [now_lr] + list(tr_log) + list(vl_log))
        colmuns = ['epoch', 'lr', 'tr_loss', 'tr_score', 'vl_loss', 'vl_score']
        save_log(loglist, colmuns, filename_head + 'training_log.csv')

    return net

def test_model(net, loader, criterion, filename_head='', use_label=False):
    net = net.cuda()

    total_loss, score, values = tester(net, loader, criterion, use_label=use_label, return_value=True)

    columns = ['pred_logit'+str(i) for i in range(values.shape[1] - 2)]
    columns = ['true_label', 'pred_label'] + columns
    save_log(values, columns, filename_head + str(score) + '_' + 'test_result.csv')

    return