import argparse
import time
import copy
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--rung',       default='A', choices=['A', 'B', 'C', 'D'])
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--warmup',     type=int, default=5,
                    help='warmup forward passes before timing')
parser.add_argument('--repeats',    type=int, default=20,
                    help='timed forward passes per measurement')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_CUDA = device.type == 'cuda'
print(f'[profile] device={device}  rung={args.rung}  '
      f'batch={args.batch_size}  warmup={args.warmup}  repeats={args.repeats}')

RUNG_CONFIG = {
    'A': dict(n_req=5,  n_uav=2,  n_adr=2, n_depots_uav=1, n_depots_adr=1),
    'B': dict(n_req=10, n_uav=4,  n_adr=3, n_depots_uav=1, n_depots_adr=1),
    'C': dict(n_req=25, n_uav=5,  n_adr=4, n_depots_uav=2, n_depots_adr=2),
    'D': dict(n_req=60, n_uav=10, n_adr=8, n_depots_uav=3, n_depots_adr=2),
}

cfg         = RUNG_CONFIG[args.rung]
n_req       = cfg['n_req']
n_uav       = cfg['n_uav']
n_adr       = cfg['n_adr']
n_depots    = cfg['n_depots_uav'] + cfg['n_depots_adr']
n_agents    = n_uav + n_adr
steps       = n_depots + n_req * 2

from creat_vrp import creat_data

print('[profile] generating one batch…', flush=True)
loader = creat_data(
    n_req, n_uav, n_adr, cfg['n_depots_uav'], cfg['n_depots_adr'],
    num_samples=args.batch_size,
    batch_size=args.batch_size,
    streaming=False,
    use_cache=False,
    seed=99,
    num_workers=0,
    distributed=False,
)
batch = next(iter(loader))
batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
print(f'[profile] batch ready — n_total nodes = {batch["x"].shape[1]}', flush=True)

from VRP_Actor import Model
from rolloutBaseline1 import RolloutBaseline

model = Model(
    input_node_dim=11,
    hidden_node_dim=128,
    input_edge_dim=4,
    hidden_edge_dim=16,
    conv_layers=3,
).to(device).eval()

bl_loader = [batch]
baseline = RolloutBaseline(model, bl_loader, n_uav, n_adr, n_nodes=n_req, epoch=0)

def _time_fn(fn, n_warmup, n_repeat):
    for _ in range(n_warmup):
        with torch.no_grad():
            fn()
    if USE_CUDA:
        torch.cuda.synchronize()

    times = []
    if USE_CUDA:
        for _ in range(n_repeat):
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end   = torch.cuda.Event(enable_timing=True)
            ev_start.record()
            with torch.no_grad():
                fn()
            ev_end.record()
            torch.cuda.synchronize()
            times.append(ev_start.elapsed_time(ev_end))
    else:
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            with torch.no_grad():
                fn()
            times.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(times)), float(np.std(times))

print('\n[profile] warming up and timing…', flush=True)

def full_train():
    model(batch, n_uav, n_adr, greedy=False, parallel_select=False)

t_full_seq_mean, t_full_seq_std = _time_fn(full_train, args.warmup, args.repeats)

def full_greedy_seq():
    model(batch, n_uav, n_adr, greedy=True, parallel_select=False)

t_greedy_seq_mean, t_greedy_seq_std = _time_fn(full_greedy_seq, args.warmup, args.repeats)

def full_greedy_par():
    model(batch, n_uav, n_adr, greedy=True, parallel_select=True)

t_greedy_par_mean, t_greedy_par_std = _time_fn(full_greedy_par, args.warmup, args.repeats)

def encoder_only():
    model.encoder(batch)

t_enc_mean, t_enc_std = _time_fn(encoder_only, args.warmup, args.repeats)

def bl_eval():
    baseline.eval(batch, n_uav, n_adr, steps)

t_bl_mean, t_bl_std = _time_fn(bl_eval, args.warmup, args.repeats)

t_dec_seq_mean = max(t_full_seq_mean - t_enc_mean, 0.0)
t_dec_par_mean = max(t_greedy_par_mean - t_enc_mean, 0.0)

enc_pct_seq = 100.0 * t_enc_mean / max(t_full_seq_mean, 1e-9)
enc_pct_par = 100.0 * t_enc_mean / max(t_greedy_par_mean, 1e-9)
speedup_par  = t_greedy_seq_mean / max(t_greedy_par_mean, 1e-9)

W = 55
print('\n' + '=' * W)
print(f'  Timing report — Rung {args.rung} | batch={args.batch_size} | '
      f'agents={n_agents} | steps≤{steps}')
print('=' * W)

rows = [
    ('Full fwd (train, seq decoder)',   t_full_seq_mean,   t_full_seq_std),
    ('Full fwd (greedy eval, seq)',      t_greedy_seq_mean, t_greedy_seq_std),
    ('Full fwd (greedy eval, parallel)', t_greedy_par_mean, t_greedy_par_std),
    ('Encoder only',                     t_enc_mean,        t_enc_std),
    ('Decoder sequential (derived)',     t_dec_seq_mean,    0.0),
    ('Decoder parallel   (derived)',     t_dec_par_mean,    0.0),
    ('Baseline.eval (frozen greedy)',    t_bl_mean,         t_bl_std),
]
for label, mean, std in rows:
    print(f'  {label:<40s}  {mean:7.2f} ± {std:5.2f} ms')

print('-' * W)
print(f'  Encoder share of seq forward:   {enc_pct_seq:5.1f}%')
print(f'  Encoder share of par forward:   {enc_pct_par:5.1f}%')
print(f'  Parallel-select speedup (eval): {speedup_par:5.2f}×')
print('=' * W)

print('\n[recommendation]')
if enc_pct_seq >= 60:
    print(f'  Encoder dominates ({enc_pct_seq:.0f}% of forward pass).')
    print('  Sequential decoder loop is NOT the bottleneck.')
    print('  parallel_select gains < {:.0f}% of total forward time.'.format(
        100 - enc_pct_seq))
    print('  Focus: encoder efficiency (fewer conv layers, smaller hidden dim).')
elif speedup_par >= 1.5:
    print(f'  Decoder sequential is significant ({100 - enc_pct_seq:.0f}% of fwd).')
    print(f'  parallel_select gives {speedup_par:.2f}× speedup on eval.')
    print('  Recommendation: keep sequential for REINFORCE training (correct gradient).')
    print('  Use parallel_select=True for all inference/evaluation calls.')
    if speedup_par >= 2.5:
        print('  Training speedup possible: consider BL_EVAL_FREQ=5 first (zero quality cost).')
        print('  If still bottlenecked, parallel training with biased gradient is an option.')
else:
    print(f'  Balanced: encoder={enc_pct_seq:.0f}%, decoder={100-enc_pct_seq:.0f}%.')
    print(f'  parallel_select gives only {speedup_par:.2f}× — marginal gain.')
    print('  Keep current design (sequential train, parallel eval).')

print(f'\n  Baseline.eval cost: {t_bl_mean:.2f} ms per batch.')
if t_bl_mean > t_full_seq_mean * 0.8:
    print(f'  WARNING: baseline.eval costs ~{t_bl_mean/t_full_seq_mean:.1f}× the training fwd pass.')
    print('  Set BL_EVAL_FREQ=5 (export BL_EVAL_FREQ=5) to cut this 5×.')
else:
    print(f'  ({t_bl_mean/t_full_seq_mean:.2f}× training fwd pass — acceptable.)')
print()
