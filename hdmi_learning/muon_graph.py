"""Optional CUDA Graph acceleration for Muon Newton-Schulz updates.

Uses a private PyTorch API, validated with torch 2.11. Installation is process-wide
and should happen once per training worker. Outputs are reused on the next call
and must be consumed immediately on the calling stream, as in MuonAdamWWrapper.
CPU, non-FP32 inputs and enclosing CUDA graph captures retain eager behavior.
"""
import torch
import torch.optim._muon as muon

_original = muon._zeropower_via_newtonschulz
_cache = {}

def graph_ns(grad, ns_coefficients, ns_steps, eps):
    # FP32 parameters produce FP32 update tensors in this workload. Other dtypes
    # retain eager behavior (BF16 eager input may be modified in-place).
    if grad.device.type != "cuda" or grad.dtype != torch.float32 or torch.cuda.is_current_stream_capturing():
        return _original(grad, ns_coefficients, ns_steps, eps)
    key = (torch.cuda.current_stream(grad.device).cuda_stream, grad.device.index, tuple(grad.shape), tuple(grad.stride()), tuple(ns_coefficients), ns_steps, eps)
    if key not in _cache:
        current = torch.cuda.current_stream(grad.device)
        side = torch.cuda.Stream(device=grad.device)
        side.wait_stream(current)
        with torch.cuda.stream(side):
            x = torch.empty_like(grad)
            x.copy_(grad)
            for _ in range(3):
                _original(x, ns_coefficients, ns_steps, eps)
        current.wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            y = _original(x, ns_coefficients, ns_steps, eps)
        current.wait_stream(side)
        _cache[key] = (x, y, graph)
    x, y, graph = _cache[key]
    x.copy_(grad)
    graph.replay()
    return y

def install():
    """Enable for this worker; repeated calls are harmless."""
    muon._zeropower_via_newtonschulz = graph_ns
