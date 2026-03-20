#!/usr/bin/env python3
"""
Local smoke tests for leak-free control-tensor TTT.

Run with:
    python3 ttt_local_test.py
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TinyBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_scale = nn.Parameter(torch.ones(dim))
        self.mlp_scale = nn.Parameter(torch.ones(dim))
        self.resid_mix = nn.Parameter(torch.stack([torch.ones(dim), torch.zeros(dim)]))

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix
        x = mix[0] * x + mix[1] * x0
        x = x + self.attn_scale * self.fc(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp_scale * torch.relu(self.proj(F.rms_norm(x, (x.size(-1),)))).square()
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int = 64, dim: int = 32, num_layers: int = 3):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([TinyBlock(dim) for _ in range(num_layers)])
        self.dim = dim

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x0 = F.rms_norm(x, (x.size(-1),))
        for block in self.blocks:
            x = block(x, x0)
        x = F.rms_norm(x, (x.size(-1),))
        logits = F.linear(x.reshape(-1, self.dim), self.tok_emb.weight)
        return F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="mean")


CONTROL_PATTERNS = ["attn_scale", "mlp_scale", "resid_mix"]


def find_doc_boundaries(tokens: Tensor, bos_id: int = 1) -> list[tuple[int, int]]:
    positions = (tokens == bos_id).nonzero(as_tuple=True)[0].tolist()
    docs = []
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(tokens)
        if end - start >= 2:
            docs.append((start, end))
    return docs


def get_ctrl_params(model: nn.Module) -> list[nn.Parameter]:
    return [p for name, p in model.named_parameters() if any(pattern in name for pattern in CONTROL_PATTERNS)]


def save_params(params: list[nn.Parameter]) -> list[Tensor]:
    return [p.detach().clone() for p in params]


def restore_params(params: list[nn.Parameter], snapshots: list[Tensor]) -> None:
    with torch.no_grad():
        for param, snapshot in zip(params, snapshots, strict=True):
            param.copy_(snapshot)


def adapt_on_prefix(model: nn.Module, ctrl_params: list[nn.Parameter], doc: Tensor, prefix_len: int, steps: int, lr: float) -> None:
    x = doc[:prefix_len].unsqueeze(0)
    y = doc[1:prefix_len + 1].unsqueeze(0)
    for param in model.parameters():
        param.requires_grad_(False)
    for param in ctrl_params:
        param.requires_grad_(True)
        param.grad = None
    model.train()
    for _ in range(steps):
        loss = model(x, y)
        loss.backward()
        with torch.no_grad():
            for param in ctrl_params:
                if param.grad is not None:
                    param.add_(param.grad, alpha=-lr)
                    param.grad = None
    for param in ctrl_params:
        param.requires_grad_(False)


def baseline_eval(model: nn.Module, val_tokens: Tensor) -> tuple[float, int]:
    total_loss = 0.0
    total_tokens = 0
    for start, end in find_doc_boundaries(val_tokens):
        doc = val_tokens[start:end]
        if doc.numel() < 2:
            continue
        n = doc.numel() - 1
        with torch.no_grad():
            total_loss += model(doc[:-1].unsqueeze(0), doc[1:].unsqueeze(0)).item() * n
        total_tokens += n
    return total_loss / total_tokens, total_tokens


def leak_free_eval(
    model: nn.Module,
    val_tokens: Tensor,
    prefix_len: int = 10,
    steps: int = 3,
    lr: float = 0.005,
    min_doc_tokens: int = 10,
) -> tuple[float, int, dict[str, float]]:
    ctrl_params = get_ctrl_params(model)
    snapshots = save_params(ctrl_params)
    total_loss = 0.0
    total_tokens = 0
    trace: dict[str, float] = {}

    for doc_index, (start, end) in enumerate(find_doc_boundaries(val_tokens)):
        doc = val_tokens[start:end]
        doc_len = doc.numel()
        if doc_len < 2:
            continue
        do_ttt = doc_len >= min_doc_tokens
        cur_prefix = min(prefix_len, doc_len - 1) if do_ttt else doc_len - 1

        model.eval()
        with torch.no_grad():
            if cur_prefix > 0:
                prefix_loss = model(doc[:cur_prefix].unsqueeze(0), doc[1:cur_prefix + 1].unsqueeze(0)).item()
                total_loss += prefix_loss * cur_prefix
                total_tokens += cur_prefix
                if doc_index == 0:
                    trace["prefix_loss_before"] = prefix_loss

        if do_ttt and cur_prefix > 0:
            adapt_on_prefix(model, ctrl_params, doc, cur_prefix, steps, lr)

        remainder = doc_len - 1 - cur_prefix
        if remainder > 0:
            model.eval()
            with torch.no_grad():
                remainder_loss = model(doc[cur_prefix:-1].unsqueeze(0), doc[cur_prefix + 1:].unsqueeze(0)).item()
                total_loss += remainder_loss * remainder
                total_tokens += remainder
                if doc_index == 0:
                    trace["remainder_loss_after"] = remainder_loss

        restore_params(ctrl_params, snapshots)

    return total_loss / total_tokens, total_tokens, trace


def cheating_eval(model: nn.Module, doc: Tensor, prefix_len: int = 20, steps: int = 5, lr: float = 0.005) -> float:
    ctrl_params = get_ctrl_params(model)
    snapshots = save_params(ctrl_params)
    adapt_on_prefix(model, ctrl_params, doc, prefix_len, steps, lr)
    with torch.no_grad():
        loss = model(doc[:-1].unsqueeze(0), doc[1:].unsqueeze(0)).item()
    restore_params(ctrl_params, snapshots)
    return loss


def test_document_boundaries() -> None:
    tokens = torch.tensor([1, 10, 20, 30, 1, 40, 50, 1, 60])
    docs = find_doc_boundaries(tokens)
    assert docs == [(0, 4), (4, 7), (7, 9)]


def test_adaptation_and_restore() -> None:
    torch.manual_seed(42)
    model = TinyGPT()
    ctrl_params = get_ctrl_params(model)
    snapshots = save_params(ctrl_params)
    doc = torch.randint(2, 64, (50,))
    x = doc[:-1].unsqueeze(0)
    y = doc[1:].unsqueeze(0)
    with torch.no_grad():
        before = model(x, y).item()
    adapt_on_prefix(model, ctrl_params, doc, prefix_len=25, steps=5, lr=0.01)
    with torch.no_grad():
        after = model(x, y).item()
    assert after < before
    restore_params(ctrl_params, snapshots)
    with torch.no_grad():
        restored = model(x, y).item()
    assert abs(restored - before) < 1e-5


def test_leak_free_eval() -> None:
    torch.manual_seed(42)
    model = TinyGPT()
    docs = []
    for i in range(5):
        doc_len = 20 + i * 8
        docs.append(torch.cat([torch.tensor([1]), torch.randint(2, 64, (doc_len,))]))
    val_tokens = torch.cat(docs)

    base_loss, base_tokens = baseline_eval(model, val_tokens)
    honest_loss, honest_tokens, trace = leak_free_eval(model, val_tokens)

    first_doc = val_tokens[slice(*find_doc_boundaries(val_tokens)[0])]
    with torch.no_grad():
        manual_prefix_loss = model(first_doc[:10].unsqueeze(0), first_doc[1:11].unsqueeze(0)).item()

    assert honest_tokens == base_tokens
    assert abs(trace["prefix_loss_before"] - manual_prefix_loss) < 1e-6
    assert honest_loss < base_loss


def test_no_tokens_skipped() -> None:
    torch.manual_seed(42)
    val_tokens = torch.cat([
        torch.tensor([1]), torch.randint(2, 64, (30,)),
        torch.tensor([1]), torch.randint(2, 64, (15,)),
        torch.tensor([1]), torch.randint(2, 64, (8,)),
    ])
    expected = sum(end - start - 1 for start, end in find_doc_boundaries(val_tokens))
    _, actual, _ = leak_free_eval(val_tokens=val_tokens, model=TinyGPT())
    assert actual == expected


def test_leak_free_vs_cheating() -> None:
    torch.manual_seed(42)
    model = TinyGPT()
    doc = torch.cat([torch.tensor([1]), torch.randint(2, 64, (60,))])
    with torch.no_grad():
        base_loss = model(doc[:-1].unsqueeze(0), doc[1:].unsqueeze(0)).item()
        honest_prefix = model(doc[:20].unsqueeze(0), doc[1:21].unsqueeze(0)).item()
    ctrl_params = get_ctrl_params(model)
    snapshots = save_params(ctrl_params)
    adapt_on_prefix(model, ctrl_params, doc, prefix_len=20, steps=5, lr=0.005)
    with torch.no_grad():
        honest_remainder = model(doc[20:-1].unsqueeze(0), doc[21:].unsqueeze(0)).item()
    restore_params(ctrl_params, snapshots)
    honest_loss = (honest_prefix * 20 + honest_remainder * (len(doc) - 21)) / (len(doc) - 1)
    cheat_loss = cheating_eval(model, doc)
    assert honest_loss < base_loss
    assert cheat_loss <= honest_loss + 1e-6


if __name__ == "__main__":
    test_document_boundaries()
    test_adaptation_and_restore()
    test_leak_free_eval()
    test_no_tokens_skipped()
    test_leak_free_vs_cheating()
    print("ttt_local_test.py: all tests passed")
