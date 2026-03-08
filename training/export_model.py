"""
Export trained DominoNet checkpoint to ONNX for browser inference.

Usage:
  python export_model.py training/checkpoints/domino_gen_0084.pt --output domino_model.onnx

The ONNX model can then be loaded in the browser using ONNX Runtime Web.
"""

import argparse
import json
import struct
import sys
import os

import torch
import numpy as np

# Add parent dir to path so we can import training modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from domino_net import DominoNet


def filtered_inference_state_dict(model):
    """Strip training-only belief head weights — browser format stays unchanged."""
    sd = model.state_dict()
    return {
        k: v for k, v in sd.items()
        if not k.startswith("belief_fc1")
        and not k.startswith("belief_bn")
        and not k.startswith("belief_fc2")
    }


def export_onnx(checkpoint_path, output_path):
    """Export checkpoint to ONNX format."""
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = DominoNet(input_dim=213, hidden_dim=256, num_actions=57, num_blocks=4)
    incompat = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if incompat.missing_keys:    print(f"[export] Missing keys: {incompat.missing_keys}")
    if incompat.unexpected_keys: print(f"[export] Unexpected keys: {incompat.unexpected_keys}")
    model.eval()

    print(f"Loaded checkpoint: generation {ckpt.get('generation', '?')}")
    print(f"  Buffer size: {ckpt.get('buffer_size', '?')}")

    # Dummy inputs for tracing
    dummy_state = torch.randn(1, 213)
    dummy_mask = torch.ones(1, 57)

    # Export
    torch.onnx.export(
        model,
        (dummy_state, dummy_mask),
        output_path,
        input_names=['state', 'mask'],
        output_names=['policy', 'value'],
        dynamic_axes={
            'state': {0: 'batch'},
            'mask': {0: 'batch'},
            'policy': {0: 'batch'},
            'value': {0: 'batch'},
        },
        opset_version=17,
    )
    size_kb = os.path.getsize(output_path) / 1024
    print(f"Exported ONNX model: {output_path} ({size_kb:.0f} KB)")


def export_raw_weights(checkpoint_path, output_path):
    """Export checkpoint as raw JSON weights for lightweight browser inference.

    This avoids the ONNX runtime dependency (~2MB) by extracting raw weight
    matrices that can be loaded with a tiny custom inference engine in JS.
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = DominoNet(input_dim=213, hidden_dim=256, num_actions=57, num_blocks=4)
    incompat = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if incompat.missing_keys:    print(f"[export] Missing keys: {incompat.missing_keys}")
    if incompat.unexpected_keys: print(f"[export] Unexpected keys: {incompat.unexpected_keys}")
    model.eval()

    print(f"Loaded checkpoint: generation {ckpt.get('generation', '?')}")

    weights = {}
    for name, param in filtered_inference_state_dict(model).items():
        arr = param.cpu().numpy()
        weights[name] = {
            'shape': list(arr.shape),
            'data': arr.flatten().tolist()
        }

    # Also export running mean/var for BatchNorm layers
    with open(output_path, 'w') as f:
        json.dump({
            'generation': ckpt.get('generation', 0),
            'architecture': {
                'input_dim': 213,
                'hidden_dim': 256,
                'num_actions': 57,
                'num_blocks': 4,
            },
            'weights': weights,
        }, f)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"Exported raw weights: {output_path} ({size_kb:.0f} KB)")


def export_binary_weights(checkpoint_path, output_path):
    """Export checkpoint as compact binary format for fast browser loading.

    Format: header (JSON) + binary float32 arrays.
    Much smaller than JSON (~1/3 size) and faster to parse.
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = DominoNet(input_dim=213, hidden_dim=256, num_actions=57, num_blocks=4)
    incompat = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if incompat.missing_keys:    print(f"[export] Missing keys: {incompat.missing_keys}")
    if incompat.unexpected_keys: print(f"[export] Unexpected keys: {incompat.unexpected_keys}")
    model.eval()

    print(f"Loaded checkpoint: generation {ckpt.get('generation', '?')}")

    # Collect all weight tensors in order
    layer_info = []
    all_data = []
    offset = 0
    for name, param in filtered_inference_state_dict(model).items():
        arr = param.cpu().numpy().astype(np.float32).flatten()
        layer_info.append({
            'name': name,
            'shape': list(param.shape),
            'offset': offset,
            'length': len(arr),
        })
        all_data.append(arr)
        offset += len(arr)

    header = json.dumps({
        'generation': ckpt.get('generation', 0),
        'architecture': {
            'input_dim': 213,
            'hidden_dim': 256,
            'num_actions': 57,
            'num_blocks': 4,
        },
        'layers': layer_info,
        'total_floats': offset,
    }).encode('utf-8')

    with open(output_path, 'wb') as f:
        # 4 bytes: header length
        f.write(struct.pack('<I', len(header)))
        # Header JSON
        f.write(header)
        # All float32 data concatenated
        for arr in all_data:
            f.write(arr.tobytes())

    size_kb = os.path.getsize(output_path) / 1024
    print(f"Exported binary weights: {output_path} ({size_kb:.0f} KB)")
    print(f"  Total parameters: {offset:,}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export DominoNet to browser format')
    parser.add_argument('checkpoint', help='Path to .pt checkpoint')
    parser.add_argument('--output', '-o', default='domino_model.bin', help='Output path')
    parser.add_argument('--format', choices=['onnx', 'json', 'binary'], default='binary',
                        help='Export format (default: binary)')
    args = parser.parse_args()

    if args.format == 'onnx':
        export_onnx(args.checkpoint, args.output if args.output != 'domino_model.bin' else 'domino_model.onnx')
    elif args.format == 'json':
        export_raw_weights(args.checkpoint, args.output if args.output != 'domino_model.bin' else 'domino_model.json')
    else:
        export_binary_weights(args.checkpoint, args.output)
