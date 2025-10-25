import torch
import torch.nn as nn
import numpy as np
import time
import os
import json
import yaml
from pathlib import Path
from contextlib import nullcontext
from tqdm import tqdm
import matplotlib.pyplot as plt

from .model import GPT, GPTConfig
from .dataset import ShakespeareDataset, get_batch
from .tokenizer import get_tokenizer


class Trainer:
    def __init__(self, config_path: str = None, **kwargs):
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.config = {**config, **kwargs}
        else:
            self.config = kwargs

        self.set_defaults()
        self.setup_device()
        self.setup_paths()

    def set_defaults(self):
        defaults = {
            'out_dir': 'checkpoints',
            'eval_interval': 250,
            'log_interval': 10,
            'eval_iters': 50,
            'eval_only': False,
            'always_save_checkpoint': False,
            'init_from': 'scratch',
            'dataset': 'shakespeare',
            'gradient_accumulation_steps': 1,
            'batch_size': 12,
            'block_size': 256,
            'n_layer': 6,
            'n_head': 6,
            'n_embd': 384,
            'dropout': 0.2,
            'bias': False,
            'learning_rate': 3e-4,
            'max_iters': 5000,
            'weight_decay': 1e-1,
            'beta1': 0.9,
            'beta2': 0.95,
            'grad_clip': 1.0,
            'decay_lr': True,
            'warmup_iters': 100,
            'lr_decay_iters': 5000,
            'min_lr': 6e-5,
            'seed': 1337,
            'tokenizer_type': 'character',
            'compile': False
        }
        for k, v in defaults.items():
            if k not in self.config:
                self.config[k] = v

    def setup_device(self):
        if torch.backends.mps.is_available():
            self.device = 'mps'
            print("Using MPS device")
        elif torch.cuda.is_available():
            self.device = 'cuda'
            print("Using CUDA device")
        else:
            self.device = 'cpu'
            print("Using CPU device")

        self.dtype = torch.float32
        self.ctx = nullcontext()

        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])

    def setup_paths(self):
        self.out_dir = Path(self.config['out_dir'])
        self.out_dir.mkdir(exist_ok=True, parents=True)

        self.log_file = self.out_dir / 'training_log.json'
        self.loss_plot_file = self.out_dir / 'loss_plot.png'

    def load_data(self):
        print(f"Loading {self.config['dataset']} dataset...")

        if self.config['dataset'] == 'shakespeare':
            dataset = ShakespeareDataset(
                data_dir='data',
                block_size=self.config['block_size']
            )
            self.tokenizer = get_tokenizer(self.config['tokenizer_type'])
            train_data, val_data, self.tokenizer = dataset.load_and_split(self.tokenizer)

            self.train_data = train_data
            self.val_data = val_data
            self.vocab_size = self.tokenizer.vocab_size

            tokenizer_path = self.out_dir / 'tokenizer.json'
            if hasattr(self.tokenizer, 'save'):
                self.tokenizer.save(str(tokenizer_path))
                print(f"Tokenizer saved to {tokenizer_path}")

    def init_model(self):
        model_args = dict(
            n_layer=self.config['n_layer'],
            n_head=self.config['n_head'],
            n_embd=self.config['n_embd'],
            block_size=self.config['block_size'],
            bias=self.config['bias'],
            vocab_size=self.vocab_size,
            dropout=self.config['dropout'],
            device=self.device
        )

        if self.config['init_from'] == 'scratch':
            print("Initializing model from scratch")
            gptconf = GPTConfig(**model_args)
            self.model = GPT(gptconf)
        elif self.config['init_from'] == 'resume':
            print(f"Resuming from {self.out_dir}")
            ckpt_path = self.out_dir / 'ckpt.pt'
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            gptconf = GPTConfig(**checkpoint['model_args'])
            self.model = GPT(gptconf)
            state_dict = checkpoint['model']
            self.model.load_state_dict(state_dict)
            self.iter_num = checkpoint['iter_num']
            self.best_val_loss = checkpoint['best_val_loss']

        self.model.to(self.device)

        if self.config['compile'] and self.device != 'mps':
            print("Compiling model...")
            self.model = torch.compile(self.model)

        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == torch.float16))
        self.optimizer = self.model.configure_optimizers(
            self.config['weight_decay'],
            self.config['learning_rate'],
            (self.config['beta1'], self.config['beta2']),
            self.device
        )

        if self.config['init_from'] == 'resume' and 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        checkpoint = None

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.config['eval_iters'])
            data = self.train_data if split == 'train' else self.val_data
            for k in range(self.config['eval_iters']):
                X, Y = get_batch(data, self.config['batch_size'], self.config['block_size'], self.device)
                with self.ctx:
                    logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def get_lr(self, it):
        if it < self.config['warmup_iters']:
            return self.config['learning_rate'] * it / self.config['warmup_iters']

        if it > self.config['lr_decay_iters']:
            return self.config['min_lr']

        decay_ratio = (it - self.config['warmup_iters']) / (self.config['lr_decay_iters'] - self.config['warmup_iters'])
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
        return self.config['min_lr'] + coeff * (self.config['learning_rate'] - self.config['min_lr'])

    def save_checkpoint(self, iter_num, val_loss):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'model_args': {
                'n_layer': self.config['n_layer'],
                'n_head': self.config['n_head'],
                'n_embd': self.config['n_embd'],
                'block_size': self.config['block_size'],
                'bias': self.config['bias'],
                'vocab_size': self.vocab_size,
                'dropout': self.config['dropout'],
            },
            'iter_num': iter_num,
            'best_val_loss': val_loss,
            'config': self.config,
        }
        torch.save(checkpoint, self.out_dir / 'ckpt.pt')
        print(f"Saved checkpoint at iteration {iter_num}")

    def train(self):
        self.load_data()
        self.init_model()

        self.iter_num = 0
        self.best_val_loss = 1e9
        self.loss_history = {'train': [], 'val': [], 'iters': []}

        X, Y = get_batch(self.train_data, self.config['batch_size'], self.config['block_size'], self.device)

        t0 = time.time()
        local_iter_num = 0
        running_mfu = -1.0

        print(f"Starting training for {self.config['max_iters']} iterations")

        while self.iter_num < self.config['max_iters']:

            lr = self.get_lr(self.iter_num) if self.config['decay_lr'] else self.config['learning_rate']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            if self.iter_num % self.config['eval_interval'] == 0:
                losses = self.estimate_loss()
                self.loss_history['train'].append(losses['train'])
                self.loss_history['val'].append(losses['val'])
                self.loss_history['iters'].append(self.iter_num)

                print(f"Step {self.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

                if losses['val'] < self.best_val_loss or self.config['always_save_checkpoint']:
                    self.best_val_loss = losses['val']
                    if self.iter_num > 0:
                        self.save_checkpoint(self.iter_num, losses['val'])

                self.plot_losses()

            for micro_step in range(self.config['gradient_accumulation_steps']):
                with self.ctx:
                    logits, loss = self.model(X, Y)
                    loss = loss / self.config['gradient_accumulation_steps']

                X, Y = get_batch(self.train_data, self.config['batch_size'], self.config['block_size'], self.device)
                self.scaler.scale(loss).backward()

            if self.config['grad_clip'] != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if self.iter_num % self.config['log_interval'] == 0:
                lossf = loss.item() * self.config['gradient_accumulation_steps']
                print(f"Iter {self.iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

                if self.iter_num % 100 == 0:
                    self.generate_sample()

            self.iter_num += 1
            local_iter_num += 1

        print(f"Training completed!")
        self.save_checkpoint(self.iter_num, self.best_val_loss)
        self.save_training_log()

    def plot_losses(self):
        if len(self.loss_history['iters']) > 1:
            plt.figure(figsize=(10, 6))
            plt.plot(self.loss_history['iters'], self.loss_history['train'], label='Train Loss')
            plt.plot(self.loss_history['iters'], self.loss_history['val'], label='Val Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Training Progress')
            plt.legend()
            plt.grid(True)
            plt.savefig(self.loss_plot_file)
            plt.close()

    def save_training_log(self):
        # Convert tensors to float for JSON serialization
        loss_history_json = {
            'train': [float(x) if torch.is_tensor(x) else x for x in self.loss_history['train']],
            'val': [float(x) if torch.is_tensor(x) else x for x in self.loss_history['val']],
            'iters': self.loss_history['iters']
        }
        log_data = {
            'config': self.config,
            'final_train_loss': float(loss_history_json['train'][-1]) if loss_history_json['train'] else None,
            'final_val_loss': float(loss_history_json['val'][-1]) if loss_history_json['val'] else None,
            'best_val_loss': float(self.best_val_loss),
            'total_iterations': self.iter_num,
            'loss_history': loss_history_json
        }
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

    @torch.no_grad()
    def generate_sample(self):
        self.model.eval()
        start_ids = [0]
        x = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]
        generated = self.model.generate(x, max_new_tokens=100, temperature=0.8, top_k=40)
        text = self.tokenizer.decode(generated[0].tolist())
        print(f"Sample: {text[:100]}...")
        self.model.train()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--device', type=str, default=None, help='Device to use')
    parser.add_argument('--max_iters', type=int, default=None, help='Maximum iterations')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--compile', action='store_true', help='Compile model')
    args = parser.parse_args()

    kwargs = {}
    if args.device:
        kwargs['device'] = args.device
    if args.max_iters:
        kwargs['max_iters'] = args.max_iters
    if args.batch_size:
        kwargs['batch_size'] = args.batch_size
    if args.compile:
        kwargs['compile'] = True

    trainer = Trainer(config_path=args.config, **kwargs)
    trainer.train()


if __name__ == "__main__":
    main()