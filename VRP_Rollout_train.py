import csv
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
import contextlib
from collections import OrderedDict, namedtuple
from itertools import product

from VRP_Actor import Model
from creat_vrp import creat_data, reward1, build_dynamic_masked_batch
from rolloutBaseline1 import RolloutBaseline

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

IS_DIST    = 'RANK' in os.environ and int(os.environ.get('WORLD_SIZE', 1)) > 1
RANK       = int(os.environ.get('RANK', 0))
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))

if IS_DIST:
    if not torch.cuda.is_available():
        raise RuntimeError(
            f'Distributed launch requested (WORLD_SIZE={WORLD_SIZE}) '
            f'but no CUDA GPUs are visible on this rank (LOCAL_RANK={LOCAL_RANK}). '
            f'Check that the cuda module is loaded and the job was allocated GPUs.'
        )
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group(backend='nccl', device_id=torch.device(f'cuda:{LOCAL_RANK}'))

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f'cuda:{LOCAL_RANK}')
    try:
        import torch_directml
        _dml = torch_directml.device()
        _src = torch.zeros(5, 4, device=_dml)
        _idx = torch.zeros(5, dtype=torch.long, device=_dml)
        torch.zeros(3, 4, device=_dml).scatter_add_(
            0, _idx.unsqueeze(1).expand_as(_src), _src
        )
        if RANK == 0:
            print('[device] DirectML active — Windows GPU via DirectX 12.')
            print('         Note: num_workers must be 0 and pin_memory=False on DirectML.')
        return _dml
    except Exception:
        if RANK == 0:
            print('[device] DirectML scatter_add_ probe failed — falling back to CPU.')
            print('         Training is slower on CPU but fully correct.')
    return torch.device('cpu')

device = _get_device()
_is_directml = 'privateuseone' in str(device)

if RANK == 0:
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f'[device] {device}  |  world={WORLD_SIZE}  |  CUDA GPUs={n_gpus}')

def _inner(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, 'module') else model

RUNG = 'A'

RUNG_ORDER = ['A', 'B', 'C', 'D']

RUNG_CONFIG = {
    'A': dict(n_req=5,  n_uav=2,  n_adr=2, n_depots_uav=1, n_depots_adr=1,
              train_size=1_280_000, batch_size=512, val_size=10_240, streaming=True,
              grad_accum=1, conv_layers=3),
    'B': dict(n_req=10, n_uav=4,  n_adr=3, n_depots_uav=1, n_depots_adr=1,
              train_size=1_280_000, batch_size=512, val_size=10_240, streaming=True,
              grad_accum=1, conv_layers=3),
    'C': dict(n_req=25, n_uav=5,  n_adr=4, n_depots_uav=2, n_depots_adr=2,
              train_size=640_000,   batch_size=256, val_size=5_120,  streaming=True,
              grad_accum=2, conv_layers=3),
    'D': dict(n_req=60, n_uav=10, n_adr=8, n_depots_uav=3, n_depots_adr=3,
              train_size=320_000,   batch_size=256, val_size=3_200,  streaming=True,
              grad_accum=4, conv_layers=3),
}

import argparse as _argparse
_cli = _argparse.ArgumentParser(add_help=False)
_cli.add_argument('--rung',   default=None, choices=['A', 'B', 'C', 'D'])
_cli.add_argument('--epochs', type=int, default=None)
_cli.add_argument('--smoke',  action='store_true',
                  help='CPU smoke-test: tiny data, 5 epochs, validates full pipeline')
_cli.add_argument('--overfit', action='store_true',
                  help='Overfit diagnostic: fix one batch, train on it only. '
                       'train_reward_mean in the CSV should drop steadily if the '
                       'architecture can learn. Runs alongside normal val logging.')
_cli.add_argument('--rloo-k', type=int, default=0,
                  help='RLOO trajectories per instance (0 = rollout baseline). '
                       'Recommended: 4-8. Replaces the frozen rollout baseline with '
                       'a leave-one-out estimate — much lower variance, no t-test lag.')
_cli.add_argument('--variant', default='',
                  help='Suffix appended to the checkpoint folder and CSV log name '
                       '(e.g. "rloo4") so parallel ablation runs stay separate. '
                       'Set automatically by submit_cc.sh when RLOO_K>0.')
_cli.add_argument('--warmstart', action='store_true',
                  help='Warm-start from the previous rung best_actor.pt (curriculum). '
                       'Off by default so each rung trains independently. '
                       'Enable for the full A→B→C→D curriculum chain.')
_cli.add_argument('--arch', default='hetgat', choices=['hetgat', 'simplegat'],
                  help='hetgat = full 4D edge features (default). '
                       'simplegat = distance-only 1D edge features — ablation baseline '
                       'showing the value of signed-margin / energy / temporal-gap signals.')
_cli.add_argument('--shadow-val-size', type=int, default=256,
                  help='Instances per shadow seed (used for multi-seed val tracking). '
                       'Kept small so seeds can be spaced by this value, ensuring each '
                       'seed generates a fully non-overlapping set of instances. '
                       'The primary val loader (for the rollout baseline t-test) keeps '
                       'its full val_size. Default: 256.')
_cli.add_argument('--val-seeds', type=int, nargs='+',
                  default=list(range(50_000, 50_000 + 20 * 256, 256)),
                  help='Validation seeds for multi-seed shadow plot. Each seed generates '
                       '--shadow-val-size instances (non-overlapping since seeds are spaced '
                       'by shadow-val-size). val_cost = mean, val_cost_std = std across seeds. '
                       'Default: 20 seeds starting at 50000, spaced by 256 — identical across '
                       'all baseline runs for fair comparison.')
_cli.add_argument('--hard-val-seeds', type=int, nargs='+',
                  default=list(range(60_000, 60_000 + 50 * 256, 256)),
                  help='Seeds for tight-TW hard validation sets. '
                       'Default: 50 seeds starting at 60000, spaced by 256.')
_cli.add_argument('--entropy-coef', type=float, default=0.01,
                  help='Entropy regularisation coefficient. Higher values keep exploration '
                       'alive longer and prevent premature policy collapse. '
                       'Default 0.01 (previously 0.005, raised after observing plateau at ~epoch 10).')
_cli.add_argument('--lr', type=float, default=1e-4,
                  help='Base learning rate before DDP sqrt-scaling and warmup. '
                       'Use 7e-5 for RLOO k>=8 to prevent LR-induced divergence at epoch 15+.')
_cli.add_argument('--lr-decay', type=float, default=0.97,
                  help='Multiplicative LR decay per epoch after warmup. '
                       '0.97 (default) halves the LR every ~23 epochs. '
                       '0.99 (old default) was too slow — LR stayed at 82%% of peak at epoch 20, '
                       'causing k=8 RLOO divergence.')
_cli.add_argument('--warmup-epochs', type=int, default=5,
                  help='Linear LR warmup length in epochs.')
_cli.add_argument('--dynamic', action='store_true',
                  help='On-demand training: each batch randomly reveals a fraction of '
                       'orders (n_vis ~ U[n_req//4, n_req]) so the model learns to plan '
                       'any visible subset. Required for epoch-based rolling-horizon eval.')
_cli.add_argument('--mix-hard-frac', type=float, default=0.3,
                  help='Fraction of streaming training instances that are non-uniform. '
                       'Each flag (temporal_peaks, spatial_cluster_k, demand_heavy_tail) '
                       'fires independently with this probability per instance, and TW '
                       'tightens proportionally to how many flags fired. '
                       '0.0 = all uniform (old behaviour). Default: 0.3.')
_cli_args, _ = _cli.parse_known_args()
if _cli_args.rung is not None:
    RUNG = _cli_args.rung
_CLI_MAX_EPOCHS  = _cli_args.epochs
_SMOKE           = _cli_args.smoke
_OVERFIT         = _cli_args.overfit
_RLOO_K          = _cli_args.rloo_k
_VARIANT         = _cli_args.variant
_WARMSTART       = _cli_args.warmstart
_ARCH            = _cli_args.arch
_SHADOW_VAL_SIZE = _cli_args.shadow_val_size
_VAL_SEEDS       = _cli_args.val_seeds
_HARD_VAL_SEEDS  = _cli_args.hard_val_seeds
_EDGE_DIM        = 1 if _ARCH == 'simplegat' else 4
_ENTROPY_COEF    = _cli_args.entropy_coef
_BASE_LR         = _cli_args.lr
_LR_DECAY        = _cli_args.lr_decay
_WARMUP_EPOCHS   = _cli_args.warmup_epochs
_DYNAMIC         = _cli_args.dynamic
_MIX_HARD_FRAC   = _cli_args.mix_hard_frac

if _ARCH == 'simplegat' and not _VARIANT:
    _VARIANT = 'simplegat'
if _DYNAMIC and not _VARIANT:
    _VARIANT = 'dynamic'

if _SMOKE:
    if _cli_args.rung is None:
        RUNG = 'A'
    RUNG_CONFIG[RUNG].update(
        train_size=256, batch_size=32, val_size=64, streaming=False
    )
    _SHADOW_VAL_SIZE = min(_SHADOW_VAL_SIZE, 32)
    if _CLI_MAX_EPOCHS is None:
        _CLI_MAX_EPOCHS = 5

cfg          = RUNG_CONFIG[RUNG]
n_req        = cfg['n_req']
n_uav        = cfg['n_uav']
n_adr        = cfg['n_adr']
n_depots_uav = cfg['n_depots_uav']
n_depots_adr = cfg['n_depots_adr']
n_depots     = n_depots_uav + n_depots_adr
steps        = n_depots + n_req * 2

max_grad_norm = 2.0

_QUICK_MAX_EPOCHS = None

BASELINE_EVAL_FREQ = int(os.environ.get('BL_EVAL_FREQ', '1'))

def adv_normalize(adv: torch.Tensor) -> torch.Tensor:
    if IS_DIST:
        mean = adv.mean()
        dist.all_reduce(mean, op=dist.ReduceOp.AVG)
        var = ((adv - mean) ** 2).mean()
        dist.all_reduce(var, op=dist.ReduceOp.AVG)
        std = var.sqrt()
    else:
        mean, std = adv.mean(), adv.std()
    if std == 0 or torch.isnan(std):
        return adv - mean
    return (adv - mean) / (std + 1e-8)

def _prep_batch(batch: dict) -> dict:
    if _EDGE_DIM == 1:
        batch = dict(batch)
        batch['edge_attr_uav'] = batch['edge_attr_uav'][..., 1:2]
        batch['edge_attr_adr'] = batch['edge_attr_adr'][..., 1:2]
    return batch

class _SlicedLoader:
    def __init__(self, loader):
        self._loader = loader

    def __iter__(self):
        for batch in self._loader:
            yield _prep_batch(batch)

    def __len__(self):
        return len(self._loader)

_VAL_T_CURRENT = 60.0   # fixed snapshot for stable validation metrics

def rollout(model: nn.Module, dataset, breakdown: bool = False):
    _inner(model).eval()
    all_cost, all_travel, all_tw, all_undeliv = [], [], [], []

    for b in dataset:
        batch = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}
        initial_visited_val = None
        _n_vis_val = float(n_req)
        if _DYNAMIC:
            batch, initial_visited_val, _n_vis_val = build_dynamic_masked_batch(
                batch, n_req, n_depots, t_current=_VAL_T_CURRENT, dynamic_features=True)
            _n_vis_val = _n_vis_val.float().clamp(min=1).to(device)  # [B] float
        with torch.no_grad():
            tour_idx, _, time_tensor = _inner(model)(batch, n_uav, n_adr, greedy=True,
                                                     initial_visited=initial_visited_val)
            if breakdown:
                rew, comps = reward1(
                    batch['time_window'], tour_idx.detach(),
                    batch['edge_attr_d'], batch['edge_attr_r'],
                    time_tensor, n_uav,
                    num_depots=n_depots,
                    return_breakdown=True,
                )
                all_travel.append((comps['travel'].sum(dim=1) / _n_vis_val).cpu())
                all_tw.append((comps['tw_penalty'].sum(dim=1) / _n_vis_val).cpu())
                all_undeliv.append((comps['undeliv'].sum(dim=1) / _n_vis_val).cpu())
            else:
                rew = reward1(
                    batch['time_window'], tour_idx.detach(),
                    batch['edge_attr_d'], batch['edge_attr_r'],
                    time_tensor, n_uav,
                    num_depots=n_depots,
                )
            all_cost.append((rew.sum(dim=1) / _n_vis_val).cpu())

    costs = torch.cat(all_cost)
    if not breakdown:
        return costs
    return costs, torch.cat(all_travel), torch.cat(all_tw), torch.cat(all_undeliv)

def save_checkpoint(epoch, model, optimizer, scheduler, folder, costs):
    if RANK != 0:
        return
    torch.save({
        'epoch':               epoch,
        'model_state_dict':    _inner(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'costs':               costs,
        'rng_state':           torch.get_rng_state(),
    }, os.path.join(folder, 'checkpoint.pth'))
    print(f'Checkpoint saved — epoch {epoch}', flush=True)

def load_checkpoint(model, optimizer, scheduler, path):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    _inner(model).load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if ckpt.get('rng_state') is not None and not _is_directml:
        torch.set_rng_state(ckpt['rng_state'].cpu().byte())
    return ckpt['epoch'] + 1, ckpt['costs']

def train():
    params = OrderedDict(
        lr=[_BASE_LR],
        hidden_node_dim=[128],
        hidden_edge_dim=[16],
        entropy_coef=[_ENTROPY_COEF],
    )
    Run  = namedtuple('Run', params.keys())
    runs = [Run(*v) for v in product(*params.values())]

    _cfg = dict(RUNG_CONFIG[RUNG])
    _shadow_val_size = _SHADOW_VAL_SIZE
    if _QUICK_MAX_EPOCHS is not None:
        _cfg['train_size'] = 12800
        _cfg['val_size']   = 1280
        _cfg['batch_size'] = 64
        _cfg['streaming']  = False
        _shadow_val_size = min(_shadow_val_size, 64)

    batch_size  = _cfg['batch_size']
    train_size  = _cfg['train_size']
    val_size    = _cfg['val_size']
    streaming   = _cfg['streaming']
    grad_accum  = _cfg.get('grad_accum', 1)
    conv_layers = _cfg.get('conv_layers', 3)

    _no_workers = _is_directml or not torch.cuda.is_available()
    train_workers = 0 if _no_workers else 4
    val_workers   = 0 if _no_workers else 2

    _run_tag = f'{RUNG}-{_VARIANT}' if _VARIANT else RUNG
    folder = f'CE-PDPTW-{_run_tag}-HetGAT'
    if RANK == 0:
        os.makedirs(folder, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
    if IS_DIST:
        dist.barrier()

    csv_log_path = os.path.join('logs', f'training_{_run_tag}.csv')
    _csv_file = None
    _csv_writer = None
    _tb_writer = None
    if RANK == 0:
        _csv_fields = [
            'epoch', 'rung', 'arch', 'rloo_k', 'entropy_coef', 'lr_decay',
            'train_reward_mean', 'train_loss_mean',
            'val_cost', 'val_cost_std',
            'val_travel_cost', 'val_tw_penalty', 'val_undeliv_cost',
            'val_hard_cost', 'val_hard_tw_penalty',
            'grad_norm_mean', 'grad_norm_max',
            'baseline_updated', 'mode_uav_frac', 'lr', 'elapsed_s',
        ]
        _csv_file = open(csv_log_path, 'w', newline='')
        _csv_writer = csv.DictWriter(_csv_file, fieldnames=_csv_fields)
        _csv_writer.writeheader()
        if _TB_AVAILABLE:
            _tb_writer = SummaryWriter(log_dir=os.path.join('logs', f'tb_{RUNG}'))

    for lr, h_node, h_edge, ent_coef in runs:
        effective_lr = lr * np.sqrt(WORLD_SIZE) if IS_DIST else lr

        if RANK == 0:
            print(f'\nConfig: lr={lr} (eff={effective_lr:.2e}), bs={batch_size}×{WORLD_SIZE}, '
                  f'h={h_node}, conv={conv_layers}, rung={RUNG}', flush=True)

        cache_train = f'cache_train_{RUNG}.pkl'
        cache_valid = f'cache_valid_{RUNG}.pkl'

        train_loader = creat_data(
            n_req, n_uav, n_adr, n_depots_uav, n_depots_adr,
            train_size,
            batch_size=batch_size,
            streaming=streaming,
            use_cache=(not streaming),
            cache_file=cache_train,
            seed=42,
            num_workers=train_workers,
            distributed=IS_DIST,
            rank=RANK,
            world_size=WORLD_SIZE,
            mix_hard_frac=_MIX_HARD_FRAC if streaming else 0.0,
        )

        valid_loader = creat_data(
            n_req, n_uav, n_adr, n_depots_uav, n_depots_adr,
            val_size,
            batch_size=batch_size,
            streaming=False,
            use_cache=True,
            cache_file=cache_valid,
            seed=1000,
            shuffle=False,
            num_workers=val_workers,
            distributed=False,
        )

        # Shadow val: one small non-overlapping dataset per seed.
        # Seeds are spaced by _shadow_val_size so rng(seed+i) ranges never overlap.
        # The primary valid_loader (full val_size) is kept separate for the rollout baseline.
        _val_loaders_ms = []
        for _vs in _VAL_SEEDS:
            _shadow_vl = creat_data(
                n_req, n_uav, n_adr, n_depots_uav, n_depots_adr,
                _shadow_val_size,
                batch_size=batch_size,
                streaming=False,
                use_cache=True,
                cache_file=f'cache_shadow_{RUNG}_s{_vs}_n{_shadow_val_size}.pkl',
                seed=_vs,
                shuffle=False,
                num_workers=val_workers,
                distributed=False,
            )
            _val_loaders_ms.append(_SlicedLoader(_shadow_vl))

        _val_loaders_hard = []
        for _hs in _HARD_VAL_SEEDS:
            _vl_h = _SlicedLoader(creat_data(
                n_req, n_uav, n_adr, n_depots_uav, n_depots_adr,
                _shadow_val_size,
                batch_size=batch_size,
                streaming=False,
                use_cache=True,
                cache_file=f'cache_shadow_{RUNG}_hard_s{_hs}_n{_shadow_val_size}.pkl',
                seed=_hs,
                shuffle=False,
                num_workers=val_workers,
                distributed=False,
                tw_slack_mean=10.0,
                tw_slack_std=2.0,
                tw_slack_clip=(5.0, 20.0),
                temporal_peaks=True,
                spatial_cluster_k=3,
                demand_heavy_tail=True,
            ))
            _val_loaders_hard.append(_vl_h)

        if RANK == 0:
            print('Data ready.', flush=True)

        _overfit_batch = None
        if _OVERFIT:
            _raw = next(iter(train_loader))
            _overfit_batch = _prep_batch({k: v.to(device) for k, v in _raw.items()
                                          if isinstance(v, torch.Tensor)})
            if RANK == 0:
                print(
                    f'[OVERFIT] Batch fixed: {_overfit_batch["x"].shape[0]} instances. '
                    f'train_reward_mean should decrease each epoch if architecture works.',
                    flush=True,
                )

        actor = Model(
            input_node_dim=13 if _DYNAMIC else 11,
            hidden_node_dim=h_node,
            input_edge_dim=_EDGE_DIM,
            hidden_edge_dim=h_edge,
            conv_layers=conv_layers,
            arch=_ARCH,
        ).to(device)

        if IS_DIST:
            actor = DDP(actor, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK,
                        find_unused_parameters=False)
            if RANK == 0:
                print(f'DDP active: {WORLD_SIZE} ranks.', flush=True)
        elif torch.cuda.device_count() > 1 and not _is_directml:
            actor = nn.DataParallel(actor)
            if RANK == 0:
                print(f'DataParallel on {torch.cuda.device_count()} GPUs.', flush=True)

        def _bl_transform_fn(batch):
            masked, iv, _ = build_dynamic_masked_batch(
                batch, n_req, n_depots, t_current=_VAL_T_CURRENT, dynamic_features=True)
            return masked, iv

        baseline = RolloutBaseline(
            _inner(actor), _SlicedLoader(valid_loader),
            n_uav=n_uav, n_adr=n_adr, n_nodes=steps,
            transform_fn=_bl_transform_fn if _DYNAMIC else None,
        )

        optimizer = optim.Adam(actor.parameters(), lr=effective_lr)
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda ep: (
                min(1.0, (ep + 1) / _WARMUP_EPOCHS) * (_LR_DECAY ** max(0, ep - _WARMUP_EPOCHS))
            ),
        )

        ckpt_path = os.path.join(folder, 'checkpoint.pth')
        if os.path.exists(ckpt_path):
            start_epoch, costs = load_checkpoint(actor, optimizer, scheduler, ckpt_path)
            if RANK == 0:
                print(f'Resuming from epoch {start_epoch}', flush=True)
        else:
            start_epoch, costs = 0, []
            rung_idx = RUNG_ORDER.index(RUNG)
            if rung_idx > 0 and _WARMSTART:
                prev_folder  = f'CE-PDPTW-{RUNG_ORDER[rung_idx - 1]}-HetGAT'
                prev_best    = os.path.join(prev_folder, 'best_actor.pt')
                prev_ckpt    = os.path.join(prev_folder, 'checkpoint.pth')
                if os.path.exists(prev_best):
                    prev_sd = torch.load(prev_best, map_location=device, weights_only=False)
                    _inner(actor).load_state_dict(prev_sd, strict=False)
                    if RANK == 0:
                        print(f'Warm-started from Rung {RUNG_ORDER[rung_idx-1]} '
                              f'(best_actor.pt)', flush=True)
                elif os.path.exists(prev_ckpt):
                    prev_sd = torch.load(prev_ckpt, map_location=device, weights_only=False)['model_state_dict']
                    _inner(actor).load_state_dict(prev_sd, strict=False)
                    if RANK == 0:
                        print(f'Warm-started from Rung {RUNG_ORDER[rung_idx-1]} '
                              f'(checkpoint.pth — best_actor.pt not found)', flush=True)
            elif not _WARMSTART and RANK == 0:
                print(f'[init] Rung {RUNG} starting from random init (use --warmstart for curriculum).', flush=True)
            if IS_DIST:
                for p in _inner(actor).parameters():
                    dist.broadcast(p.data, src=0)

        if _RLOO_K == 1:
            raise ValueError(
                '--rloo-k 1 is invalid: leave-one-out baseline requires K>=2 '
                '(K=1 divides by zero). Use --rloo-k 4.'
            )

        _use_amp = torch.cuda.is_available() and not _is_directml

        max_epochs = _QUICK_MAX_EPOCHS if _QUICK_MAX_EPOCHS is not None else (_CLI_MAX_EPOCHS or 100)
        if _SMOKE and RANK == 0:
            print(
                f'\n[SMOKE] Pipeline check: Rung {RUNG}, {train_size} train, '
                f'{val_size} val, {max_epochs} epochs, device={device}',
                flush=True,
            )
        best_val_cost = min(costs) if costs else float('inf')
        for epoch in range(start_epoch, max_epochs):
            if RANK == 0:
                print(f'\nEpoch {epoch} -----------------------------------', flush=True)

            actor.train()
            losses, rewards_log, grad_norms_log = [], [], []
            uav_actions_total, all_actions_total = 0, 0
            _cached_bl_reward = None
            start = time.time()

            if not _OVERFIT and IS_DIST and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            _train_src = [_overfit_batch] if _OVERFIT else train_loader

            for batch_idx, batch in enumerate(_train_src):
                if not _OVERFIT:
                    batch = _prep_batch({k: v.to(device) for k, v in batch.items()
                                         if isinstance(v, torch.Tensor)})

                _initial_visited_train = None
                if _DYNAMIC and not _OVERFIT:
                    _t_current = float(torch.empty(1).uniform_(0.0, 120.0).item())
                    batch, _initial_visited_train, _n_vis = build_dynamic_masked_batch(
                        batch, n_req, n_depots, t_current=_t_current, dynamic_features=True)
                    _n_vis = _n_vis.float().clamp(min=1).to(device)  # [B] float
                else:
                    _n_vis = float(n_req)

                if batch_idx % grad_accum == 0:
                    optimizer.zero_grad(set_to_none=True)

                _amp_cm = (torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
                           if _use_amp else contextlib.nullcontext())

                _K = _RLOO_K if _RLOO_K > 0 else 1
                _B = batch['x'].shape[0]
                _batch_fw = (
                    {k: v.repeat_interleave(_K, dim=0) for k, v in batch.items()
                     if isinstance(v, torch.Tensor)}
                    if _RLOO_K > 0 else batch
                )
                # Expand initial_visited to match RLOO batch size
                _iv_fw = (
                    _initial_visited_train.repeat_interleave(_K, dim=0)
                    if _RLOO_K > 0 and isinstance(_initial_visited_train, torch.Tensor)
                    else _initial_visited_train
                )

                with _amp_cm:
                    tour_idx_fw, tour_logp_fw, time_tensor_fw = actor(
                        _batch_fw, n_uav, n_adr,
                        greedy=False, T=1.0,
                        checkpoint_encoder=True, training=True,
                        initial_visited=_iv_fw,
                    )

                # Per-instance cost normalization; repeat for RLOO's K-expanded batch
                _n_vis_fw = (
                    _n_vis.repeat_interleave(_K)
                    if _RLOO_K > 0 and isinstance(_n_vis, torch.Tensor)
                    else _n_vis
                )
                reward_fw = reward1(
                    _batch_fw['time_window'], tour_idx_fw.detach(),
                    _batch_fw['edge_attr_d'], _batch_fw['edge_attr_r'],
                    time_tensor_fw, n_uav,
                    num_depots=n_depots,
                ).sum(dim=1) / _n_vis_fw

                if _RLOO_K > 0:
                    reward_2d = reward_fw.view(_B, _K)
                    bl_rloo   = (reward_2d.sum(1, keepdim=True) - reward_2d) / (_K - 1)
                    advantage = adv_normalize((reward_2d - bl_rloo).view(-1))
                    reward    = reward_2d.mean(1)
                    tour_idx  = tour_idx_fw.view(_B, _K, n_uav + n_adr, -1)[:, 0]
                elif _OVERFIT:
                    reward        = reward_fw
                    baseline_reward = reward_fw.detach().mean().expand_as(reward_fw)
                    advantage     = adv_normalize(reward_fw - baseline_reward)
                    tour_idx      = tour_idx_fw
                else:
                    reward = reward_fw
                    if (_cached_bl_reward is None
                            or batch_idx % BASELINE_EVAL_FREQ == 0
                            or _cached_bl_reward.size(0) != batch['time_window'].size(0)):
                        _cached_bl_reward = baseline.eval(
                            batch, n_uav, n_adr, steps,
                            initial_visited=_initial_visited_train).detach()
                    baseline_reward = _cached_bl_reward / _n_vis
                    advantage = adv_normalize(reward_fw - baseline_reward)
                    tour_idx  = tour_idx_fw

                actual_steps = max(tour_idx_fw.shape[2] - 1, 1)
                logp_per_step = tour_logp_fw.sum(dim=1) / actual_steps
                actor_loss = torch.mean(advantage.detach() * logp_per_step)
                logp_reg   = -(logp_per_step.mean()) * ent_coef
                total_loss = actor_loss - logp_reg

                _eff_accum = 1 if _OVERFIT else grad_accum
                (total_loss / _eff_accum).backward()

                _is_step = (
                    _OVERFIT
                    or (batch_idx + 1) % grad_accum == 0
                    or (batch_idx + 1) == len(train_loader)
                )
                if _is_step:
                    grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                    optimizer.step()
                else:
                    grad_norm = torch.tensor(0.0)

                rewards_log.append(reward.mean().item())
                losses.append(actor_loss.item())
                grad_norms_log.append(grad_norm.item())

                with torch.no_grad():
                    batch_size_b = tour_idx.size(0)
                    actions_b = tour_idx[:, :, 1:]
                    is_customer = actions_b >= n_depots
                    uav_actions_total += int(is_customer[:, :n_uav, :].sum().item())
                    all_actions_total += int(is_customer.sum().item())

                if RANK == 0 and (batch_idx + 1) % 50 == 0:
                    print(
                        f'  [{batch_idx+1}/{len(train_loader)}] '
                        f'rew={np.mean(rewards_log[-50:]):.4f}  '
                        f'loss={np.mean(losses[-50:]):.4f}  '
                        f'gnorm={np.mean(grad_norms_log[-50:]):.3f}  '
                        f'elapsed={time.time()-start:.0f}s',
                        flush=True,
                    )

            scheduler.step()

            if _OVERFIT or _RLOO_K > 0 or epoch < _WARMUP_EPOCHS:
                baseline_updated = 0
            elif not IS_DIST:
                baseline.epoch_callback(_inner(actor), epoch)
                baseline_updated = int(getattr(baseline, 'last_update_accepted', False))
            else:
                _upd_flag = torch.zeros(1, dtype=torch.long, device=device)
                _bl_mean  = torch.zeros(1, device=device)
                if RANK == 0:
                    baseline.epoch_callback(_inner(actor), epoch)
                    _upd_flag[0] = int(getattr(baseline, 'last_update_accepted', False))
                    _bl_mean[0]  = baseline.mean
                dist.broadcast(_upd_flag, src=0)
                dist.broadcast(_bl_mean,  src=0)
                baseline_updated = int(_upd_flag.item())
                if baseline_updated:
                    for _p in baseline.model.parameters():
                        dist.broadcast(_p.data, src=0)
                    for _b in baseline.model.buffers():
                        dist.broadcast(_b, src=0)
                baseline.mean = float(_bl_mean.item())

            if RANK == 0 and (epoch % 25 == 0 or epoch == max_epochs - 1):
                epoch_dir = os.path.join(folder, str(epoch))
                os.makedirs(epoch_dir, exist_ok=True)
                torch.save(_inner(actor).state_dict(),
                           os.path.join(epoch_dir, 'actor.pt'))

            _seed_costs = torch.zeros(len(_val_loaders_ms), device=device)
            _breakdown_vals = torch.zeros(5, device=device)
            if not IS_DIST or RANK == 0:
                for _si, _vl in enumerate(_val_loaders_ms):
                    if _si == 0:
                        _sc, _tr, _tw, _ud = rollout(_inner(actor), _vl, breakdown=True)
                        _seed_costs[0] = _sc.mean()
                        _breakdown_vals[0] = _tr.mean()
                        _breakdown_vals[1] = _tw.mean()
                        _breakdown_vals[2] = _ud.mean()
                    else:
                        _seed_costs[_si] = rollout(_inner(actor), _vl).mean()
                _hard_cost_vals, _hard_tw_vals = [], []
                for _vl_h in _val_loaders_hard:
                    _hc, _, _htw, _ = rollout(_inner(actor), _vl_h, breakdown=True)
                    _hard_cost_vals.append(_hc.mean().item())
                    _hard_tw_vals.append(_htw.mean().item())
                _breakdown_vals[3] = sum(_hard_cost_vals) / len(_hard_cost_vals)
                _breakdown_vals[4] = sum(_hard_tw_vals)  / len(_hard_tw_vals)
            if IS_DIST:
                dist.broadcast(_seed_costs,     src=0)
                dist.broadcast(_breakdown_vals, src=0)
            val_cost          = float(_seed_costs.mean().item())
            val_cost_std      = float(_seed_costs.std().item()) if len(_val_loaders_ms) > 1 else 0.0
            val_travel_cost   = float(_breakdown_vals[0])
            val_tw_penalty    = float(_breakdown_vals[1])
            val_undeliv_cost  = float(_breakdown_vals[2])
            val_hard_cost     = float(_breakdown_vals[3])
            val_hard_tw_penalty = float(_breakdown_vals[4])

            if RANK == 0:
                costs.append(val_cost)
                np.savetxt(os.path.join(folder, 'costs.txt'), costs)
                save_checkpoint(epoch, actor, optimizer, scheduler, folder, costs)

                if val_cost < best_val_cost:
                    best_val_cost = val_cost
                    torch.save(_inner(actor).state_dict(),
                               os.path.join(folder, 'best_actor.pt'))
                    print(f'  New best val cost: {val_cost:.4f} -> saved best_actor.pt',
                          flush=True)

                mode_uav_frac = (uav_actions_total / max(all_actions_total, 1))
                current_lr = optimizer.param_groups[0]['lr']
                elapsed = time.time() - start
                grad_norm_max = float(max(grad_norms_log)) if grad_norms_log else 0.0

                _std_str = f' ±{val_cost_std:.4f}' if val_cost_std > 0 else ''
                print(
                    f'  Rung {RUNG} | {_ARCH} | Epoch {epoch} | '
                    f'Val cost: {val_cost:.4f}{_std_str} | '
                    f'[travel={val_travel_cost:.4f} tw={val_tw_penalty:.4f} ud={val_undeliv_cost:.4f}] | '
                    f'Hard: cost={val_hard_cost:.4f} tw={val_hard_tw_penalty:.4f} | '
                    f'UAV frac: {mode_uav_frac:.3f} | '
                    f'grad_norm: {np.mean(grad_norms_log):.3f} (max {grad_norm_max:.3f}) | '
                    f'bl_updated: {bool(baseline_updated)}',
                    flush=True,
                )
                print(f'  History: {[f"{c:.3f}" for c in costs[-5:]]}', flush=True)

                if _csv_writer is not None:
                    _csv_writer.writerow({
                        'epoch':              epoch,
                        'rung':               RUNG,
                        'arch':               _ARCH,
                        'rloo_k':             _RLOO_K,
                        'entropy_coef':       _ENTROPY_COEF,
                        'lr_decay':           _LR_DECAY,
                        'train_reward_mean':  np.mean(rewards_log),
                        'train_loss_mean':    np.mean(losses),
                        'val_cost':           val_cost,
                        'val_cost_std':       val_cost_std,
                        'val_travel_cost':    val_travel_cost,
                        'val_tw_penalty':     val_tw_penalty,
                        'val_undeliv_cost':   val_undeliv_cost,
                        'val_hard_cost':      val_hard_cost,
                        'val_hard_tw_penalty': val_hard_tw_penalty,
                        'grad_norm_mean':     np.mean(grad_norms_log),
                        'grad_norm_max':      grad_norm_max,
                        'baseline_updated':   baseline_updated,
                        'mode_uav_frac':      mode_uav_frac,
                        'lr':                 current_lr,
                        'elapsed_s':          elapsed,
                    })
                    _csv_file.flush()

                if _tb_writer is not None:
                    step = epoch
                    _tb_writer.add_scalar('train/reward_mean', np.mean(rewards_log), step)
                    _tb_writer.add_scalar('train/loss_mean', np.mean(losses), step)
                    _tb_writer.add_scalar('train/grad_norm_mean', np.mean(grad_norms_log), step)
                    _tb_writer.add_scalar('train/grad_norm_max', grad_norm_max, step)
                    _tb_writer.add_scalar('train/baseline_updated', baseline_updated, step)
                    _tb_writer.add_scalar('val/cost', val_cost, step)
                    _tb_writer.add_scalar('train/mode_uav_frac', mode_uav_frac, step)
                    _tb_writer.add_scalar('train/lr', current_lr, step)

        if _SMOKE and RANK == 0:
            first_cost, last_cost = costs[0], costs[-1]
            cost_improved = last_cost <= first_cost
            checkpoint_ok = os.path.exists(os.path.join(folder, 'checkpoint.pth'))
            best_ok       = os.path.exists(os.path.join(folder, 'best_actor.pt'))
            print('\n' + '=' * 55)
            print('[SMOKE] Pipeline verdict')
            print('=' * 55)
            print(f'  Val cost trajectory: {[f"{c:.4f}" for c in costs]}')
            print(f'  Cost improved epoch 0->last: {"YES" if cost_improved else "NO - may need more epochs"}')
            print(f'  Checkpoint saved:    {"YES" if checkpoint_ok else "FAIL"}')
            print(f'  best_actor.pt saved: {"YES" if best_ok else "FAIL"}')
            print(f'  CSV log exists:      {"YES" if os.path.exists(csv_log_path) else "FAIL"}')
            print('  Ready to submit to cluster.' if (checkpoint_ok and best_ok)
                  else '  Fix failures before submitting.')
            print('=' * 55)

        if IS_DIST:
            dist.barrier()

    if RANK == 0:
        if _csv_file is not None:
            _csv_file.close()
        if _tb_writer is not None:
            _tb_writer.close()

if __name__ == '__main__':
    try:
        train()
    finally:
        if IS_DIST:
            dist.destroy_process_group()
