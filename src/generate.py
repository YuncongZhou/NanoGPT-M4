#!/usr/bin/env python3
import torch
import argparse
import json
from pathlib import Path
from .model import GPT, GPTConfig
from .tokenizer import get_tokenizer, CharacterTokenizer


class Generator:
    def __init__(self, checkpoint_path: str, device: str = None):
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_dir = self.checkpoint_path.parent if self.checkpoint_path.is_file() else self.checkpoint_path

        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.load_model()
        self.load_tokenizer()

    def load_model(self):
        ckpt_file = self.checkpoint_path if self.checkpoint_path.is_file() else self.checkpoint_dir / 'ckpt.pt'

        print(f"Loading model from {ckpt_file}")
        checkpoint = torch.load(ckpt_file, map_location=self.device)

        model_args = checkpoint['model_args']
        model_args['device'] = self.device

        gptconf = GPTConfig(**model_args)
        self.model = GPT(gptconf)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.model.to(self.device)

        self.config = checkpoint.get('config', {})
        print(f"Model loaded successfully! ({self.model.get_num_params()/1e6:.2f}M parameters)")

    def load_tokenizer(self):
        tokenizer_path = self.checkpoint_dir / 'tokenizer.json'
        if tokenizer_path.exists():
            self.tokenizer = CharacterTokenizer.load(str(tokenizer_path))
            print(f"Loaded tokenizer with vocab size {self.tokenizer.vocab_size}")
        else:
            print("Warning: No tokenizer found, using default character tokenizer")
            self.tokenizer = get_tokenizer('character')

    @torch.no_grad()
    def generate(
        self,
        prompt: str = "",
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 200,
        top_p: float = None,
        num_samples: int = 1,
        seed: int = None
    ):
        if seed is not None:
            torch.manual_seed(seed)

        if prompt:
            ids = self.tokenizer.encode(prompt)
        else:
            ids = [0]

        x = torch.tensor(ids, dtype=torch.long, device=self.device)[None, ...]

        samples = []
        for _ in range(num_samples):
            y = self.model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            generated_text = self.tokenizer.decode(y[0].tolist())
            samples.append(generated_text)

        return samples

    def interactive(self):
        print("\n" + "=" * 60)
        print("Interactive Text Generation")
        print("Commands: 'quit' to exit, 'reset' to clear context")
        print("=" * 60)

        context = ""

        while True:
            prompt = input("\nPrompt > ").strip()

            if prompt.lower() == 'quit':
                break
            elif prompt.lower() == 'reset':
                context = ""
                print("Context cleared.")
                continue

            if prompt:
                context = prompt

            print("\nGenerating...\n")
            samples = self.generate(
                prompt=context,
                max_new_tokens=200,
                temperature=0.8,
                top_k=40
            )

            for i, sample in enumerate(samples):
                print(f"{sample}\n")

    def batch_generate(self, prompts: list, **kwargs):
        results = []
        for prompt in prompts:
            samples = self.generate(prompt=prompt, **kwargs)
            results.append({
                'prompt': prompt,
                'generated': samples
            })
        return results


def main():
    parser = argparse.ArgumentParser(description='Generate text using trained GPT model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file or directory')
    parser.add_argument('--prompt', type=str, default="", help='Input prompt')
    parser.add_argument('--max_tokens', type=int, default=100, help='Maximum new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k sampling')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--device', type=str, default=None, help='Device to use (mps/cuda/cpu)')

    args = parser.parse_args()

    generator = Generator(args.checkpoint, device=args.device)

    if args.interactive:
        generator.interactive()
    else:
        samples = generator.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            num_samples=args.num_samples,
            seed=args.seed
        )

        for i, sample in enumerate(samples):
            if args.num_samples > 1:
                print(f"\n--- Sample {i+1} ---")
            print(sample)


if __name__ == "__main__":
    main()