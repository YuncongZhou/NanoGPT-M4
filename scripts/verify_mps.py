#!/usr/bin/env python3
import torch
import platform
import psutil
import sys
import json
from datetime import datetime

def get_system_info():
    info = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cpu_count": psutil.cpu_count(),
        "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "available_memory_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built()
    }

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        test_tensor = torch.randn(1000, 1000).to(device)
        info["mps_test"] = "SUCCESS"
        info["test_computation"] = float(torch.sum(test_tensor).cpu())
    else:
        info["mps_test"] = "FAILED"
        info["fallback_device"] = "cpu"

    return info

def main():
    print("=" * 60)
    print("MPS and System Verification Report")
    print("=" * 60)

    info = get_system_info()

    print(f"Platform: {info['platform']}")
    print(f"Processor: {info['processor']}")
    print(f"Python: {info['python_version'].split()[0]}")
    print(f"PyTorch: {info['torch_version']}")
    print(f"CPU Cores: {info['cpu_count']}")
    print(f"Total Memory: {info['total_memory_gb']} GB")
    print(f"Available Memory: {info['available_memory_gb']} GB")
    print(f"MPS Available: {info['mps_available']}")
    print(f"MPS Built: {info['mps_built']}")

    if info['mps_available']:
        print(f"MPS Test: {info['mps_test']}")
        print(f"Test Computation Result: {info['test_computation']:.4f}")
        print("\n✅ MPS is available and working correctly!")
    else:
        print(f"\n⚠️  MPS is not available, will use CPU fallback")

    with open('/Users/alexzhou/nanogpt-m4/logs/system_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\nSystem info saved to logs/system_info.json")
    print("=" * 60)

if __name__ == "__main__":
    main()