"""Side-effect-free capture of native Sim-DETR transformer outputs."""

from __future__ import annotations


class TransformerOutputCapture:
    """Capture ``(hs, references, memory, saliency_scores)`` from a transformer.

    The hook does not modify the transformer's output or its state dict.  The
    one-shot ``consume`` API also makes stale values from an earlier forward
    impossible to use silently.
    """

    def __init__(self, transformer):
        self.hs = None
        self.memory = None
        self.handle = transformer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if not isinstance(output, (tuple, list)) or len(output) < 3:
            raise RuntimeError(
                "Sim-DETR transformer output must contain hs, references, memory"
            )
        self.hs = output[0]
        self.memory = output[2]

    def consume(self):
        if self.hs is None or self.memory is None:
            raise RuntimeError("No transformer output is available to consume")
        hs, memory = self.hs, self.memory
        self.hs = None
        self.memory = None
        return hs, memory

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def __del__(self):
        self.close()
