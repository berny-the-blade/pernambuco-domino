"""
GPU Inference Server for Pernambuco Domino self-play.

Runs as a daemon process. Workers send (state, mask) pairs via a shared
request queue; results are returned via per-worker response queues.

Request format:  (worker_id, request_id, state_np, mask_np)
Response format: (request_id, policy_np, value_float)

Collects requests into mini-batches of up to `batch_size` (default 16),
or flushes after `timeout_ms` (default 2 ms), then runs a single GPU
forward pass and distributes results.
"""

import os
import sys
import time
import numpy as np
import torch

# Allow imports from training/ when running standalone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from domino_net import DominoNet


class GPUInferenceServer:
    """
    Batched GPU inference server shared by all self-play workers.

    Args:
        model_state_dict: CPU state dict to load onto GPU.
        request_queue:    mp.Queue that workers push requests onto.
        response_queues:  list of mp.Queue, one per worker (indexed by worker_id).
        batch_size:       Maximum requests to batch together (default 16).
        timeout_ms:       Max milliseconds to wait for a full batch (default 2).
    """

    def __init__(self, model_state_dict, request_queue, response_queues,
                 batch_size=16, timeout_ms=2):
        self.model_state_dict = model_state_dict
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms

    # ------------------------------------------------------------------
    # Main loop (run this method as the body of a daemon mp.Process)
    # ------------------------------------------------------------------

    def run(self):
        """Load model on GPU and serve batched inference requests forever."""
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPUInferenceServer requires CUDA, but no GPU was found."
            )

        device = torch.device("cuda")
        model = DominoNet().to(device)
        model.load_state_dict(self.model_state_dict)
        model.eval()

        print(
            f"[GPU Server] Running on {torch.cuda.get_device_name(0)}, "
            f"batch_size={self.batch_size}, timeout={self.timeout_ms}ms",
            flush=True,
        )

        with torch.no_grad():
            while True:
                batch = self._collect_batch()
                if not batch:
                    # No requests arrived within the idle timeout — loop.
                    continue

                # ── Stack numpy arrays into GPU tensors ──────────────────
                states = torch.tensor(
                    np.stack([r[2] for r in batch]),
                    dtype=torch.float32,
                    device=device,
                )
                masks = torch.tensor(
                    np.stack([r[3] for r in batch]),
                    dtype=torch.float32,
                    device=device,
                )

                # ── Single batched forward pass ───────────────────────────
                policies, values = model(states, valid_actions_mask=masks)
                policies_np = policies.cpu().numpy()  # (B, 57)
                values_np = values.cpu().numpy()      # (B, 1)

                # ── Distribute results to per-worker response queues ──────
                for i, (worker_id, request_id, _, _) in enumerate(batch):
                    self.response_queues[worker_id].put(
                        (request_id, policies_np[i], float(values_np[i, 0]))
                    )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_batch(self):
        """
        Block until at least one request arrives, then collect additional
        requests until batch_size is reached or timeout_ms elapses.

        Returns a (possibly empty) list of request tuples.
        """
        batch = []

        # Block until first item (idle timeout = 1 s so we don't spin)
        try:
            item = self.request_queue.get(timeout=1.0)
            batch.append(item)
        except Exception:
            return batch  # timed out with nothing — caller loops

        # Greedily collect more items within the window
        deadline = time.monotonic() + self.timeout_ms / 1000.0
        while len(batch) < self.batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = self.request_queue.get(timeout=max(1e-4, remaining))
                batch.append(item)
            except Exception:
                break

        return batch


# ---------------------------------------------------------------------------
# Standalone sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Importing GPUInferenceServer... OK")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("GPUInferenceServer class is ready.")
