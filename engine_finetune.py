import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Optional
from timm.data import Mixup
from timm.utils import accuracy
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, average_precision_score,
    hamming_loss, jaccard_score, recall_score, precision_score, cohen_kappa_score, r2_score, mean_absolute_error
)
from pycm import ConfusionMatrix
import util.misc as misc
import util.lr_sched as lr_sched

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    problem: str,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None
):
    """Train the model for one epoch."""
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print_freq, accum_iter = 20, args.accum_iter
    optimizer.zero_grad()
    
    if log_writer:
        print(f'log_dir: {log_writer.log_dir}')
    
    for data_iter_step, (samples, targets, _) in enumerate(metric_logger.log_every(data_loader, print_freq, f'Epoch: [{epoch}]')):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        samples, targets = samples.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        if mixup_fn:
            samples, targets = mixup_fn(samples, targets)
        
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs.squeeze(), targets)
        loss_value = loss.item()
        loss /= accum_iter
        
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss/train', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, criterion, problem, device, args, epoch, mode, num_class, log_writer):
    """Evaluate the model."""
    metric_logger = misc.MetricLogger(delimiter="  ")
    os.makedirs(os.path.join(args.output_dir, args.task), exist_ok=True)
    
    model.eval()
    true_onehot, pred_onehot, true_labels, pred_labels, pred_softmax = [], [], [], [], []
    
    for batch in metric_logger.log_every(data_loader, 10, f'{mode}:'):
        images, target = batch[0].to(device, non_blocking=True), torch.Tensor(batch[1]).to(device, non_blocking=True)
        # target_onehot = F.one_hot(target.to(torch.int64), num_classes=num_class)
        
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output.squeeze(), target)

        if problem == 'classification':
            output = nn.Softmax(dim=1)(output)
            output = output.argmax(dim=1)
            # output_onehot = F.one_hot(output_label.to(torch.int64), num_classes=num_class)    
        
        metric_logger.update(loss=loss.item())
        # true_onehot.extend(target_onehot.cpu().numpy())
        # pred_onehot.extend(output_onehot.detach().cpu().numpy())
        true_labels.extend(target.cpu().numpy())
        pred_labels.extend(output.detach().cpu().numpy())
        # pred_softmax.extend(output_.detach().cpu().numpy())
    
    if problem == 'classification':
        accuracy = accuracy_score(true_labels, pred_labels)
        hamming = hamming_loss(true_labels, pred_labels)
        jaccard = jaccard_score(true_labels, pred_labels, average='macro')
        # average_precision = average_precision_score(true_labels, pred_softmax, average='macro')
        kappa = cohen_kappa_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, zero_division=0, average='binary', pos_label=1)
        roc_auc = roc_auc_score(true_labels, pred_labels, multi_class='ovr', average='macro')
        precision = precision_score(true_labels, pred_labels, zero_division=0, average='macro')
        recall = recall_score(true_labels, pred_labels, zero_division=0, average='binary', pos_label=1)
        specificity = recall_score(true_labels, pred_labels, zero_division=0, average='binary', pos_label=0)
        # score = (f1 + roc_auc + kappa) / 3
        score = f1
    else:
        r2 = r2_score(true_labels, pred_labels)
        mae = mean_absolute_error(true_labels, pred_labels)
        score = mae
    
    
    if log_writer:
        if problem == 'classification':
            for metric_name, value in zip(['accuracy', 'f1', 'roc_auc', 'hamming', 'jaccard', 'precision', 'recall', 'kappa', 'score'],
                                        [accuracy, f1, roc_auc, hamming, jaccard, precision, recall, kappa, score]):
                log_writer.add_scalar(f'perf/{metric_name}', value, epoch)
            print(f'Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Sensitivity: {recall:.4f}, Specificity: {specificity:.4f}')
        else:
            for metric_name, value in zip(['mae', 'r2'],
                                        [mae, r2]):
                log_writer.add_scalar(f'perf/{metric_name}', value, epoch)
            print(f'MAE: {mae:.4f}, R2: {r2:.4f}')
    
    print(f'val loss: {metric_logger.meters["loss"].global_avg}')
    
    metric_logger.synchronize_between_processes()
    
    results_path = os.path.join(args.output_dir, args.task, f'metrics_{mode}.csv')
    file_exists = os.path.isfile(results_path)
    with open(results_path, 'a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        if not file_exists:
            if problem == 'classification':
                wf.writerow(['val_loss', 'accuracy', 'f1', 'roc_auc', 'hamming', 'jaccard', 'precision', 'recall', 'kappa'])
                wf.writerow([metric_logger.meters["loss"].global_avg, accuracy, f1, roc_auc, hamming, jaccard, precision, recall, kappa])
            else:
                wf.writerow([metric_logger.meters["loss"].global_avg, mae, r2])
                wf.writerow(['val_loss', 'mae', 'r2'])
    
    if mode == 'test':
        cm = ConfusionMatrix(actual_vector=true_labels, predict_vector=pred_labels)
        cm.plot(cmap=plt.cm.Blues, number_label=True, normalized=True, plot_lib="matplotlib")
        plt.savefig(os.path.join(args.output_dir, args.task, 'confusion_matrix_test.jpg'), dpi=600, bbox_inches='tight')
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, score
