"""Exercise the actual V15 revision backward on the requested runtime device.

Runs before expensive calibration/training. No files or subprocesses, no change
to the caller's model, RNG streams or deterministic/TF32/cuDNN settings.
Compatible with Python 3.8 and PyTorch 1.9.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
# Direct script execution puts tools/v15 first, where our profile.py would
# shadow Python's stdlib profile during torch.optim's cProfile import.
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path[:] = [item for item in sys.path
               if not item or Path(item).resolve() not in (SCRIPT_DIRECTORY, ROOT)]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import torch

from model.observation_revision import ObservationRevision


def _determinism(enabled, warn_only=False):
    # warn_only is not a keyword of use_deterministic_algorithms on torch 1.9.
    if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled"):
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    else:
        torch.use_deterministic_algorithms(enabled)


def _finite(tensor, name):
    if not bool(torch.isfinite(tensor).all()):
        raise RuntimeError("nonfinite " + name)


def check_revision_backward(device):
    """Return a small report after strict FP32 forward/backward/optimizer checks.

    CUDA uses the selected device, never silently falls back to CPU. All work
    uses a temporary module. The nonzero output projection ensures the first
    backward reaches interpolation, local attention and both encoders instead
    of passing only because the production module starts as an identity.
    """
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("V15 runtime check supports CPU or CUDA, got %s" % device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("V15 runtime check requires the requested CUDA device; CUDA is unavailable")
        index = torch.cuda.current_device() if device.index is None else device.index
        device = torch.device("cuda", index)
        devices = [index]
    else:
        devices = []
    warn_getter = getattr(torch, "is_deterministic_algorithms_warn_only_enabled", None)
    previous = {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": bool(warn_getter()) if warn_getter is not None else False,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
    }
    report = None
    try:
        _determinism(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.random.fork_rng(devices=devices), torch.enable_grad():
            # Seed only the CPU global generator used by module constructors.
            # torch.manual_seed() also seeds OTHER CUDA devices, so avoid it.
            torch.random.default_generator.manual_seed(15037)
            generator = torch.Generator(device="cpu").manual_seed(15038)
            channels, height, width = 12, 16, 20
            sensor = (13, 15)  # partial patches, a fully padded column, and padded pixels
            model = ObservationRevision(channels=channels, width=12, heads=3,
                                        patch=4, radius=2, pair_batch=1,
                                        source="observation", sensor_hw=sensor)
            model = model.to(device=device, dtype=torch.float32).train()
            with torch.no_grad():
                values = torch.linspace(-0.2, 0.3, model.output.weight.numel(),
                                        device=device, dtype=torch.float32)
                model.output.weight.copy_(values.reshape_as(model.output.weight))
            initial_output = model.output.weight.detach().clone()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            shape = (2, channels, height, width)
            samples = [torch.randn(shape, generator=generator, dtype=torch.float32)
                       for _ in range(4)]
            losses, gradient_norms = [], []
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                old, first, future = [value.to(device).clone().requires_grad_(True)
                                      for value in samples[:3]]
                target = samples[3].to(device)
                revised = model.revise(old, model.encode_observation(first),
                                       model.encode_observation(future))
                _finite(revised, "revision output")
                loss = (revised - target).square().mean()
                _finite(loss, "loss")
                loss.backward()
                norms = {}
                for name, parameter in model.named_parameters():
                    if parameter.grad is None:
                        raise RuntimeError("missing parameter gradient: " + name)
                    _finite(parameter.grad, "parameter gradient " + name)
                    norms[name] = float(parameter.grad.detach().abs().sum())
                for name in ("encoder.0.weight", "query.weight", "key.weight",
                             "value.weight", "output.weight", "gate.weight"):
                    if norms[name] <= 0:
                        raise RuntimeError("inactive backward path: " + name)
                for name, value in (("old_features", old), ("old_observation", first),
                                    ("future_observation", future)):
                    if value.grad is None:
                        raise RuntimeError("missing input gradient: " + name)
                    _finite(value.grad, "input gradient " + name)
                    norms[name] = float(value.grad.detach().abs().sum())
                    if norms[name] <= 0:
                        raise RuntimeError("inactive input gradient: " + name)
                optimizer.step()
                for name, parameter in model.named_parameters():
                    _finite(parameter, "updated parameter " + name)
                losses.append(float(loss.detach()))
                gradient_norms.append(norms)
            if torch.equal(initial_output, model.output.weight.detach()):
                raise RuntimeError("optimizer did not update the nonzero output projection")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            report = {"status": "passed", "torch_version": torch.__version__,
                      "device": str(device),
                      "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                      "precision": "float32", "deterministic_algorithms": True,
                      "tf32": False, "optimizer_steps": 2, "canvas_hw": [height, width],
                      "sensor_hw": list(sensor), "nonzero_output_initialization": True,
                      "losses": losses, "gradient_l1": gradient_norms}
    except Exception as error:
        raise RuntimeError("V15 strict deterministic revision backward check failed on %s "
                           "with torch %s, before calibration/training. Do not disable "
                           "determinism to bypass this check. Original error: %s" %
                           (device, torch.__version__, error)) from error
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = previous["cudnn_tf32"]
        torch.backends.cudnn.benchmark = previous["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = previous["cudnn_deterministic"]
        _determinism(previous["deterministic"], previous["warn_only"])
    print("V15_RUNTIME_BACKWARD_OK " + json.dumps(
        {key: value for key, value in report.items() if key != "gradient_l1"}, sort_keys=True), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    check_revision_backward(args.device)


if __name__ == "__main__":
    main()
