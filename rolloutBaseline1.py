import copy
import numpy as np
import torch
from scipy.stats import ttest_rel
from torch.nn import DataParallel
from creat_vrp import reward1

def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model

def rollout1(model, dataset, num_uav, num_adr, n_nodes, transform_fn=None):
    model.eval()
    device = next(model.parameters()).device

    def eval_batch(batch):
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        initial_visited = None
        if transform_fn is not None:
            batch, initial_visited = transform_fn(batch)
        with torch.no_grad():
            tour, _, time_tensor = model(batch, num_uav, num_adr, greedy=True,
                                         initial_visited=initial_visited)
            nd = batch.get('n_depots', None)
            if nd is not None:
                num_depots = int(nd[0]) if hasattr(nd, '__len__') else int(nd)
            else:
                num_depots = None
            cost = reward1(
                batch['time_window'],
                tour.detach(),
                batch['edge_attr_d'],
                batch['edge_attr_r'],
                time_tensor,
                num_uav,
                num_depots=num_depots,
            ).sum(dim=1)
        return cost.cpu()

    return torch.cat([eval_batch(batch) for batch in dataset], dim=0)

class RolloutBaseline:
    def __init__(self, model, dataset, n_uav, n_adr, n_nodes=50, epoch=0,
                 transform_fn=None):
        self.n_nodes = n_nodes
        self.n_uav = n_uav
        self.n_adr = n_adr
        self.dataset = dataset
        self.last_update_accepted = False
        self.transform_fn = transform_fn
        self._update_model(model, epoch)

    def _update_model(self, model, epoch, dataset=None):
        if dataset is not None:
            self.dataset = dataset
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.bl_vals = rollout1(
            self.model, self.dataset, self.n_uav, self.n_adr, self.n_nodes,
            transform_fn=self.transform_fn,
        ).cpu().numpy()
        self.mean = float(self.bl_vals.mean())
        self.epoch = epoch

    def eval(self, x, num_uav, num_adr, n_nodes, initial_visited=None):
        device = next(self.model.parameters()).device
        x = {k: v.to(device) for k, v in x.items() if isinstance(v, torch.Tensor)}
        self.model.eval()
        with torch.no_grad():
            tour, _, time_tensor = self.model(x, num_uav, num_adr, greedy=True,
                                              initial_visited=initial_visited)
            nd = x.get('n_depots', None)
            if nd is not None:
                num_depots = int(nd[0]) if hasattr(nd, '__len__') else int(nd)
            else:
                num_depots = None
            value = reward1(
                x['time_window'],
                tour.detach(),
                x['edge_attr_d'],
                x['edge_attr_r'],
                time_tensor,
                num_uav,
                num_depots=num_depots,
            ).sum(dim=1)
        return value

    def epoch_callback(self, model, epoch):
        print('Evaluating candidate model on validation dataset')
        cand_vals = rollout1(
            model, self.dataset, self.n_uav, self.n_adr, self.n_nodes,
            transform_fn=self.transform_fn,
        ).cpu().numpy()

        cand_mean = float(cand_vals.mean())
        diff = cand_mean - self.mean
        print(
            f'Epoch {epoch} candidate mean {cand_mean:.4f}, '
            f'baseline epoch {self.epoch} mean {self.mean:.4f}, '
            f'diff {diff:.4f}'
        )

        if diff >= 0:
            print('Keep baseline')
            self.last_update_accepted = False
            return

        cand_episode = cand_vals
        base_episode = self.bl_vals

        if len(cand_episode) < 2:
            print('Keep baseline')
            return

        t_stat, p_two_sided = ttest_rel(cand_episode, base_episode, nan_policy='omit')
        if np.isnan(t_stat) or np.isnan(p_two_sided):
            print('Keep baseline')
            return

        p_val = p_two_sided / 2.0
        print(f'p-value: {p_val:.4f}')

        if t_stat < 0 and p_val < 0.05:
            print('Updating baseline')
            self._update_model(model, epoch)
            self.last_update_accepted = True
        else:
            print('Keep baseline')
            self.last_update_accepted = False

    def state_dict(self):
        return {
            'model': self.model,
            'dataset': self.dataset,
            'epoch': self.epoch,
        }

    def load_state_dict(self, state_dict):
        model_copy = copy.deepcopy(self.model)
        get_inner_model(model_copy).load_state_dict(
            get_inner_model(state_dict['model']).state_dict()
        )
        self._update_model(model_copy, state_dict['epoch'], state_dict['dataset'])
