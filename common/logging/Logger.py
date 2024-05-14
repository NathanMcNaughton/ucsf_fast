import torch
import numpy as np
import string
import os
import pickle

from common import lsave, lload
from common import save_model as _save_model


def _run_hash(args):
    def valid_attr(k):
        return (k not in ['proj', 'dataset', 'epochs', 'workers', 'run_id', 'scalars', 'saveparam', 'half', 'k', 'iid',
                          'hash']) and not k.startswith('Final')

    config = vars(args)
    attrs = [k for k in config.keys() if valid_attr(k)]
    return '_'.join([f'{a}-{config[a]}' for a in attrs])


class VanillaLogger():
    '''
        Logs locally instead of GCS and uses wandb for experiment tracking.
    '''

    def __init__(self, args, wandb, log_root='./logs', hash=False):
        if hash:
            args.hash = _run_hash(args)  # for easy run-grouping
        self.wandb = wandb
        proj_name = args.proj
        wandb.config.update(args)  # set wandb config to arguments
        self.run_name = wandb.run.name
        self.run_id = wandb.run.id
        args.run_name = self.run_name
        args.run_id = self.run_id

        self.logdir = os.path.join(log_root, proj_name, f"{self.run_name}-{self.run_id}")
        self.modeldir = os.path.join(self.logdir, 'models')

        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.modeldir, exist_ok=True)

        print("Logdir:", self.logdir)
        self.save(vars(args), 'config')

        self._step = 0
        self.scalars_log = []

    def save(self, obj, ext):
        lsave(obj, os.path.join(self.logdir, ext))

    def save_model(self, model):
        local_path = os.path.join(self.modeldir, 'model.pt')
        _save_model(model, local_path)

    def save_model_step(self, step, model):
        step_dir = os.path.join(self.modeldir, f'step{step:06}')
        os.makedirs(step_dir, exist_ok=True)
        local_path = os.path.join(step_dir, 'model.pt')
        save_model(model, local_path)

    def log_root(self, D: dict):
        for k, v in D.items():
            self.save(v, k)

    def log_scalars(self, D: dict, step=None, log_wandb=True):
        if step is None:
            step = self._step
            self._step += 1

        if log_wandb:
            self.wandb.log(D)
        self.scalars_log.append(D)

    def log_summary(self, D):
        self.wandb.summary = D
        self.save(D, 'summary')

    def log_step(self, step, D: dict):
        prefix = f'steps/step{step:06}'
        for k, v in D.items():
            self.save(v, f'{prefix}/{k}')

    def log_final(self, D: dict):
        prefix = 'final'
        for k, v in D.items():
            self.save(v, f'{prefix}/{k}')

    def flush(self):
        ''' saves the result of all log_scalar calls '''
        self.save(self.scalars_log, 'scalars')
