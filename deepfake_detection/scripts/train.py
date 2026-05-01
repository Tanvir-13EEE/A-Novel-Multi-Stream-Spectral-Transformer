"""
Training script.
Usage:
    python scripts/train.py
    python scripts/train.py --config configs/custom.yaml
"""

import os
import sys
import argparse
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from src.models  import MultiStreamSpectralTransformer
from src.models.components import MultiStreamFusion
from src.data    import DeepfakeDataset
from src.utils.scaler import fit_scaler


def train(cfg):
    os.makedirs(cfg['paths']['save_dir'], exist_ok=True)
    DEVICE    = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    N_WORKERS = os.cpu_count()

    scaler_path = os.path.join(
        cfg['paths']['save_dir'],
        cfg['paths']['scaler_name'])
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = fit_scaler(
            cfg['data']['root_dir'],
            cfg['data']['img_size'],
            scaler_path)

    train_ds = DeepfakeDataset(
        cfg['data']['root_dir'], 'train',
        cfg['data']['img_size'], scaler)
    val_ds   = DeepfakeDataset(
        cfg['data']['root_dir'], 'val',
        cfg['data']['img_size'], scaler)

    lkw = dict(
        batch_size         = cfg['data']['batch_size'],
        num_workers        = N_WORKERS,
        pin_memory         = (DEVICE.type=='cuda'),
        persistent_workers = True,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **lkw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **lkw)

    model = MultiStreamSpectralTransformer(
        img_size    = cfg['data']['img_size'],
        d_model     = cfg['model']['d_model'],
        num_heads   = cfg['model']['num_heads'],
        num_layers  = cfg['model']['num_layers'],
        task        = cfg['model']['task'],
        num_classes = cfg['model']['num_classes'],
    ).to(DEVICE)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer  = optim.AdamW(
        model.parameters(),
        lr           = cfg['training']['lr'],
        weight_decay = cfg['training']['weight_decay'])
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = cfg['training']['epochs'],
        eta_min = cfg['training']['eta_min'])
    criterion  = nn.CrossEntropyLoss(
        label_smoothing=cfg['training']['label_smoothing'])
    use_amp    = (DEVICE.type == 'cuda')
    scaler_amp = torch.amp.GradScaler('cuda', enabled=use_amp)

    history  = {k:[] for k in
                ['train_loss','val_loss','train_acc','val_acc']}
    best_acc  = 0.
    best_path = os.path.join(
        cfg['paths']['save_dir'],
        cfg['paths']['best_model'])

    EPOCHS = cfg['training']['epochs']
    print(f"\n{'='*55}")
    print(f" MSST | {EPOCHS} epochs | "
          f"task={cfg['model']['task']}")
    print(f"{'='*55}\n")

    for epoch in range(1, EPOCHS+1):
        model.train()
        t_loss=t_corr=t_total=0
        for imgs,phys,labels in tqdm(
                train_loader,
                desc=f"Ep {epoch:02d}/{EPOCHS} [train]",
                leave=False):
            imgs   = imgs.to(DEVICE,   non_blocking=True)
            phys   = phys.to(DEVICE,   non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, gates = model(imgs, phys)
                loss          = criterion(logits, labels)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg['training']['grad_clip'])
            scaler_amp.step(optimizer); scaler_amp.update()
            t_loss  += loss.item()
            t_corr  += (logits.argmax(1)==labels).sum().item()
            t_total += labels.size(0)
        scheduler.step()

        model.eval(); v_loss=v_corr=v_total=0
        val_gates = None
        with torch.no_grad():
            for imgs,phys,labels in tqdm(
                    val_loader,
                    desc=f"Ep {epoch:02d}/{EPOCHS} [val]  ",
                    leave=False):
                imgs   = imgs.to(DEVICE,   non_blocking=True)
                phys   = phys.to(DEVICE,   non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda',
                        enabled=use_amp):
                    logits, gates = model(imgs, phys)
                    loss          = criterion(logits, labels)
                v_loss    += loss.item()
                v_corr    += (
                    logits.argmax(1)==labels).sum().item()
                v_total   += labels.size(0)
                val_gates  = gates

        t_acc=t_corr/t_total; v_acc=v_corr/v_total
        t_l=t_loss/len(train_loader)
        v_l=v_loss/len(val_loader)
        for k,v in zip(
                ['train_loss','val_loss','train_acc','val_acc'],
                [t_l,v_l,t_acc,v_acc]):
            history[k].append(v)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"T-loss {t_l:.4f}  T-acc {t_acc:.4f} | "
              f"V-loss {v_l:.4f}  V-acc {v_acc:.4f} | "
              f"LR {scheduler.get_last_lr()[0]:.2e}")

        if (val_gates is not None and
                not val_gates.isnan().any() and
                epoch % 5 == 0):
            print("  Stream gates:")
            for name, val in zip(
                    MultiStreamFusion.STREAM_NAMES,
                    val_gates.mean(0).tolist()):
                bar = '█' * int(val * 50)
                print(f"    {name:<22} {val:.4f}  {bar}")

        if v_acc > best_acc:
            best_acc = v_acc
            m = (model.module
                 if isinstance(model, nn.DataParallel)
                 else model)
            torch.save({
                'epoch'      : epoch,
                'model_state': m.state_dict(),
                'val_acc'    : v_acc,
                'config'     : cfg,
            }, best_path)
            joblib.dump(scaler, os.path.join(
                cfg['paths']['save_dir'],
                'scaler_final.pkl'))
            print(f"  ★ Best  val_acc={best_acc:.4f}")

    # Save history
    hist_path = os.path.join(
        cfg['paths']['save_dir'],
        cfg['paths']['history'])
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Plot
    ex = range(1, EPOCHS+1)
    fig, axes = plt.subplots(1, 2, figsize=(13,5))
    axes[0].plot(ex,history['train_loss'],label='Train',lw=2)
    axes[0].plot(ex,history['val_loss'],  label='Val',  lw=2)
    axes[0].set(title='Loss',xlabel='Epoch',ylabel='Loss')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(ex,history['train_acc'],label='Train',lw=2)
    axes[1].plot(ex,history['val_acc'],  label='Val',  lw=2)
    axes[1].axhline(best_acc,color='green',ls='--',lw=1.5,
                    label=f'Best={best_acc:.4f}')
    axes[1].set(title='Accuracy',xlabel='Epoch',
                ylabel='Accuracy')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(
        cfg['paths']['save_dir'],
        cfg['paths']['plot']), dpi=150)
    plt.close()

    print(f"\n[Done] Best val accuracy: {best_acc:.4f}")
    return best_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default='configs/default.yaml')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)
