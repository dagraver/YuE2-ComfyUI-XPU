"""Whisper large-v3 turbo's network in plain torch: a 32-layer audio encoder and a 4-layer text decoder.

The shapes are those of the turbo release, and module names follow the file's
keys under ``model.``, so the weights load without a rename table.

The encoder hears thirty seconds of 128-bin log-mel: two convolutions, the
second halving, give 1500 frames, learned positions are added, and 32
pre-norm layers of 20 heads attend over all of them. In float16 every layer's
output is clamped a thousand short of the type's largest value, as
transformers does. The decoder embeds tokens and adds one of 448 learned
positions; each of its four layers attends over the tokens before it, then
over the encoder's frames, then feeds forward. Its output is read with the
token embedding.

Every attention call is the one transformers makes with its SDPA attention:
the queries scaled before the product and the call itself unscaled, no mask,
causal only for a prefill of several tokens. So a plain step picks the tokens
transformers picks. The decoder's cache is held for the whole context of 448
positions and starts as zeros; the encoder's keys and values for the decoder
sit in buffers of their own, rewritten for every window. One captured step
then serves every window and every temperature of a song (``GraphStep``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..constants import CAPTURE_MODE

MELS = 128
WIDTH = 1280
HEADS = 20
HEAD = 64
ENCODER_LAYERS = 32
DECODER_LAYERS = 4
FFN = 5120
SOURCE_POSITIONS = 1500
CONTEXT = 448
VOCAB = 51866
CLAMP = 65504.0 - 1000.0
"""float16's largest value less a thousand: where transformers clamps each encoder layer's output in float16."""
GRAPHS = True


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = nn.Linear(WIDTH, WIDTH, bias=False)
        self.v_proj = nn.Linear(WIDTH, WIDTH)
        self.q_proj = nn.Linear(WIDTH, WIDTH)
        self.out_proj = nn.Linear(WIDTH, WIDTH)

    def queries(self, x):
        """``[1, length, width]`` to ``[1, heads, length, head]``, scaled before the product as transformers scales them."""
        return (self.q_proj(x) * HEAD ** -0.5).view(1, x.shape[1], -1, HEAD).transpose(1, 2).contiguous()

    def keys_values(self, x):
        keys = self.k_proj(x).view(1, -1, HEADS, HEAD).transpose(1, 2).contiguous()
        values = self.v_proj(x).view(1, -1, HEADS, HEAD).transpose(1, 2).contiguous()
        return keys, values

    def out(self, attended, length: int):
        return self.out_proj(attended.transpose(1, 2).contiguous().reshape(1, length, -1).contiguous())


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()
        self.self_attn_layer_norm = nn.LayerNorm(WIDTH)
        self.fc1 = nn.Linear(WIDTH, FFN)
        self.fc2 = nn.Linear(FFN, WIDTH)
        self.final_layer_norm = nn.LayerNorm(WIDTH)

    def forward(self, x):
        heard = self.self_attn_layer_norm(x)
        keys, values = self.self_attn.keys_values(heard)
        attended = F.scaled_dot_product_attention(self.self_attn.queries(heard), keys, values, scale=1.0)
        x = x + self.self_attn.out(attended, x.shape[1])
        x = x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))
        if x.dtype == torch.float16:
            x = torch.clamp(x, min=-CLAMP, max=CLAMP)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(MELS, WIDTH, 3, padding=1)
        self.conv2 = nn.Conv1d(WIDTH, WIDTH, 3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(SOURCE_POSITIONS, WIDTH)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(ENCODER_LAYERS)])
        self.layer_norm = nn.LayerNorm(WIDTH)

    def forward(self, mel):
        """``[1, 128, 3000]`` log-mel in the model's type to ``[1, 1500, 1280]`` frames."""
        x = F.gelu(self.conv2(F.gelu(self.conv1(mel)))).permute(0, 2, 1)
        x = x + self.embed_positions.weight
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()
        self.self_attn_layer_norm = nn.LayerNorm(WIDTH)
        self.encoder_attn = Attention()
        self.encoder_attn_layer_norm = nn.LayerNorm(WIDTH)
        self.fc1 = nn.Linear(WIDTH, FFN)
        self.fc2 = nn.Linear(FFN, WIDTH)
        self.final_layer_norm = nn.LayerNorm(WIDTH)

    def heard(self, x, cross_keys, cross_values):
        """Cross-attention over the window's frames and the feed-forward, both after the self-attention."""
        attended = F.scaled_dot_product_attention(self.encoder_attn.queries(self.encoder_attn_layer_norm(x)),
                                                  cross_keys, cross_values, scale=1.0)
        x = x + self.encoder_attn.out(attended, x.shape[1])
        return x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, WIDTH)
        self.embed_positions = nn.Embedding(CONTEXT, WIDTH)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(DECODER_LAYERS)])
        self.layer_norm = nn.LayerNorm(WIDTH)


class Network(nn.Module):
    """The whole model. ``encode`` hears a window, ``listen`` hands it to the decoder, ``prefill`` and ``step`` decode."""

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    @property
    def device(self):
        return self.decoder.embed_tokens.weight.device

    @property
    def dtype(self):
        return self.decoder.embed_tokens.weight.dtype

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """``[128, 3000]`` float32 log-mel to the encoder's ``[1, 1500, 1280]`` frames."""
        return self.encoder(mel.to(self.device, self.dtype)[None])

    def static(self) -> dict:
        """The decoder's cache and the window's cross-attention keys and values, kept until ``forget_steps``.

        Everything starts as zeros, not as whatever the memory held: the
        captured step attends over the whole cache and hides the unfilled
        part with a mask, and an infinity left there turns the masked sum into
        NaN. Qwen3-ASR's step, left unfilled, disagreed with the plain one on
        every 190-second song with one allocator and on none with the other.
        """
        kept = getattr(self, "_static", None)
        if kept is not None and kept["device"] == self.device:
            return kept
        self.forget_steps()
        device, dtype = self.device, self.dtype

        def zeros(length):
            return torch.zeros(1, HEADS, length, HEAD, device=device, dtype=dtype)

        self._static = {"device": device, "step": None,
                        "cache": [(zeros(CONTEXT), zeros(CONTEXT)) for _ in range(DECODER_LAYERS)],
                        "cross": [(zeros(SOURCE_POSITIONS), zeros(SOURCE_POSITIONS)) for _ in range(DECODER_LAYERS)]}
        return self._static

    def forget_steps(self) -> None:
        """Let go of the kept cache, the window's buffers and the captured step."""
        self._static = None

    def listen(self, static: dict, encoded: torch.Tensor) -> None:
        """Write a window's keys and values for every decoder layer into the buffers the steps read."""
        for layer, (keys, values) in zip(self.decoder.layers, static["cross"]):
            heard_keys, heard_values = layer.encoder_attn.keys_values(encoded)
            keys.copy_(heard_keys)
            values.copy_(heard_values)

    def prefill(self, ids: list, static: dict) -> torch.Tensor:
        """Logits after each of ``ids``, ``[len(ids), vocab]`` in float32; the cache then holds their positions."""
        tokens = torch.tensor([ids], device=self.device)
        return self._decode(tokens, 0, static)

    def step(self, token: int, position: int, static: dict) -> torch.Tensor:
        """Logits after ``token`` fed at ``position``, ``[1, vocab]`` in float32, attending over the cache up to it."""
        tokens = torch.tensor([[token]], device=self.device)
        return self._decode(tokens, position, static)

    def _decode(self, tokens, position: int, static: dict):
        length = tokens.shape[1]
        if length > 1 and position:
            raise ValueError("a prefill of several tokens starts at position 0")
        end = position + length
        x = self.decoder.embed_tokens(tokens) + self.decoder.embed_positions.weight[position:end]
        for layer, (keys, values), (cross_keys, cross_values) in zip(self.decoder.layers, static["cache"],
                                                                      static["cross"]):
            heard = layer.self_attn_layer_norm(x)
            new_keys, new_values = layer.self_attn.keys_values(heard)
            keys[:, :, position:end] = new_keys
            values[:, :, position:end] = new_values
            if not position:
                attended = F.scaled_dot_product_attention(layer.self_attn.queries(heard), new_keys, new_values,
                                                          scale=1.0, is_causal=length > 1)
            else:
                attended = F.scaled_dot_product_attention(layer.self_attn.queries(heard), keys[:, :, :end],
                                                          values[:, :, :end], scale=1.0)
            x = x + layer.self_attn.out(attended, length)
            x = layer.heard(x, cross_keys, cross_values)
        x = self.decoder.layer_norm(x)
        return F.linear(x, self.decoder.embed_tokens.weight)[0].float()

    def _graph_step(self, token, position, mask, static):
        """The step ``GraphStep`` records: every shape fixed, the whole cache attended through ``mask``."""
        x = self.decoder.embed_tokens(token) + self.decoder.embed_positions.weight.index_select(0, position)
        for layer, (keys, values), (cross_keys, cross_values) in zip(self.decoder.layers, static["cache"],
                                                                      static["cross"]):
            heard = layer.self_attn_layer_norm(x)
            new_keys, new_values = layer.self_attn.keys_values(heard)
            keys.index_copy_(2, position, new_keys)
            values.index_copy_(2, position, new_values)
            attended = F.scaled_dot_product_attention(layer.self_attn.queries(heard), keys, values, attn_mask=mask,
                                                      scale=1.0)
            x = x + layer.self_attn.out(attended, 1)
            x = layer.heard(x, cross_keys, cross_values)
        x = self.decoder.layer_norm(x)
        return F.linear(x, self.decoder.embed_tokens.weight)[0].float()

    def stepper(self, static: dict, start: int):
        """The captured step, reset for a decoding whose first token goes at ``start``; None when capture fails.

        It is recorded once, right after the first window's prefill, and the
        later windows and temperatures replay the same recording: the cache
        and the buffers it reads stay where they are, and only the mask is set
        back.
        """
        try:
            if static["step"] is None:
                static["step"] = GraphStep(self, static, start)
            else:
                static["step"].reset(start)
            return static["step"]
        except Exception:
            import logging

            logging.getLogger(__name__).warning("[yue2_comfy.whisper] decoding without a CUDA graph", exc_info=True)
            static["step"] = None
            return None


class GraphStep:
    """One decoding step recorded as a CUDA graph and replayed for every token.

    A step is some hundred small kernels, and inside a ComfyUI process a
    launch costs twice what it costs in a bare one (measured for Qwen3-ASR,
    where a step went from 60 ms to a few). The token, its position and the
    mask of the cache filled so far are tensors the graph reads, written
    before each replay; the logits it writes are read after. It is recorded
    into the pack's own graph class with ``CAPTURE_MODE``, so another pack's
    memory query from a server thread cannot abort the process while it
    records, and on torch 2.13 and later the capture's extra argument goes
    through (``runtime.guarded_graph``).
    """

    def __init__(self, net: Network, static: dict, start: int):
        """Recorded right after a prefill, then replayed once (``runtime.launched``).

        The rehearsals and that replay write only position ``start``, which its
        first step writes again.
        """
        from ..runtime import guarded_graph, launched

        device, dtype = net.device, net.dtype
        self.token = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.position = torch.zeros(1, dtype=torch.long, device=device)
        self.mask = torch.full((1, 1, 1, CONTEXT), float("-inf"), device=device, dtype=dtype)
        self.reset(start)
        self.position.fill_(start)
        run = lambda: net._graph_step(self.token, self.position, self.mask, static)  # noqa: E731
        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            for _ in range(2):
                run()
        torch.cuda.current_stream(device).wait_stream(side)
        self.graph = guarded_graph()
        with torch.cuda.graph(self.graph, capture_error_mode=CAPTURE_MODE):
            self.out = run()
        launched(self.graph)

    def reset(self, start: int) -> None:
        """Ready for a decoding whose first ``start`` positions the cache already holds."""
        self.mask.fill_(float("-inf"))
        self.mask[..., :start + 1] = 0

    def __call__(self, token: int, position: int) -> torch.Tensor:
        self.token.fill_(token)
        self.position.fill_(position)
        self.mask[..., position] = 0
        self.graph.replay()
        return self.out


class steady:
    """The process-wide settings Whisper's answers depend on, held while it runs and put back after.

    Another node can leave them changed. Float32 matrix products go at full
    precision (the log-mel's filters are one, on the CPU, where 'medium' lets
    oneDNN round through bfloat16); cuDNN chooses its convolution by its
    heuristics rather than by timing, which a run with ComfyUI's '--fast
    autotune' turns on and which can pick another kernel from one run to the
    next; and float16 products keep torch's default reduction, which is what
    transformers ran with when the port was measured against it.
    """

    def __enter__(self):
        self.saved = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                      torch.backends.cudnn.benchmark,
                      torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        return self

    def __exit__(self, *exc):
        precision, tf32, benchmark, reduced = self.saved
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = reduced
        return False
