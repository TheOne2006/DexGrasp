"""Explicit, checkpointed AdamW or hidden-linear Muon + auxiliary AdamW."""
import torch
from torch import nn
from ..vendor.muon import Muon


def parameter_groups(model):
    # Only hidden nn.Linear matrices: sparse convolution kernels, input/output
    # projections, embeddings, norms and biases remain AdamW. Never flatten a
    # Minkowski kernel into an arbitrary matrix silently.
    eligible = {id(module.weight) for name, module in model.named_modules()
                if isinstance(module, nn.Linear)
                and not name.startswith(('head.', 'embed.'))}
    groups = {'muon': [], 'adamw': []}
    names = {'muon': [], 'adamw': []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        key = 'muon' if id(parameter) in eligible else 'adamw'
        groups[key].append(parameter)
        names[key].append(name)
    return groups, names


class MuonAdamW:
    def __init__(self, model, cfg, *, fused):
        groups, self.names = parameter_groups(model)
        if not all(groups.values()):
            raise ValueError('Muon comparison requires both parameter groups')
        self.muon = Muon(groups['muon'], lr=cfg['muon_lr'],
                         momentum=cfg['muon_momentum'], ns_steps=cfg['muon_ns_steps'],
                         weight_decay=cfg['weight_decay'], adjust_lr_fn='original')
        self.adamw = torch.optim.AdamW(groups['adamw'], lr=cfg['lr'],
                                      weight_decay=cfg['weight_decay'], fused=fused)

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.muon.step()
        self.adamw.step()

    def state_dict(self):
        return dict(kind='muon_adamw_v1', names=self.names,
                    muon=self.muon.state_dict(), adamw=self.adamw.state_dict())

    def load_state_dict(self, state):
        if state.get('kind') != 'muon_adamw_v1' or state['names'] != self.names:
            raise ValueError('Muon parameter grouping changed')
        self.muon.load_state_dict(state['muon'])
        self.adamw.load_state_dict(state['adamw'])


def make_optimizer(model, cfg, *, fused=False):
    if cfg['optimizer'] == 'muon':
        return MuonAdamW(model, cfg, fused=fused)
    if cfg['optimizer'] != 'adamw':
        raise ValueError('unknown optimizer')
    return torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                             lr=cfg['lr'], weight_decay=cfg['weight_decay'], fused=fused)
