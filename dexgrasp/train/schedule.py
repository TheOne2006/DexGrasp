"""Closed-form optimizer-step schedule, identical before and after resume."""
import math


def learning_rate(cfg, completed_steps, base):
    if cfg['lr_schedule'] == 'constant':
        return base
    warmup = cfg['warmup_steps']
    if completed_steps < warmup:
        return base * (cfg['warmup_start_factor'] +
                       (1-cfg['warmup_start_factor'])*completed_steps/warmup)
    fraction = min(1., (completed_steps-warmup)/(cfg['max_steps']-warmup))
    minimum = base * cfg['lr_min']/cfg['lr']
    return minimum + (base-minimum)*(1+math.cos(math.pi*fraction))/2


def apply_learning_rate(optimizer, cfg, completed_steps):
    branches = [('adamw', optimizer, cfg['lr'])]
    if cfg['optimizer'] == 'muon':
        branches = [('muon', optimizer.muon, cfg['muon_lr']),
                    ('adamw', optimizer.adamw, cfg['lr'])]
    rates = {}
    for name, branch, base in branches:
        rates[name] = learning_rate(cfg, completed_steps, base)
        for group in branch.param_groups:
            group['lr'] = rates[name]
    return rates
