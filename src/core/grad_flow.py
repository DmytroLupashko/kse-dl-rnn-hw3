import torch
import torch.nn as nn

CELL_TYPES = {nn.RNN: nn.RNNCell, nn.GRU: nn.GRUCell, nn.LSTM: nn.LSTMCell}


# ---- Copy trained nn.RNN / nn.GRU / nn.LSTM weights into a stack of cells (one per layer) ----
def build_cells_from_module(rnn_module):
    cells = nn.ModuleList()
    for l in range(rnn_module.num_layers):
        in_size = rnn_module.input_size if l == 0 else rnn_module.hidden_size
        extra = (
            {"nonlinearity": rnn_module.nonlinearity}
            if type(rnn_module) is nn.RNN
            else {}
        )
        cell = CELL_TYPES[type(rnn_module)](
            in_size, rnn_module.hidden_size, bias=rnn_module.bias, **extra
        )
        # weight_ih_l{l}, weight_hh_l{l}, bias_ih_l{l}, bias_hh_l{l}
        cell.weight_ih.data.copy_(getattr(rnn_module, f"weight_ih_l{l}").data)
        cell.weight_hh.data.copy_(getattr(rnn_module, f"weight_hh_l{l}").data)
        if rnn_module.bias:
            cell.bias_ih.data.copy_(getattr(rnn_module, f"bias_ih_l{l}").data)
            cell.bias_hh.data.copy_(getattr(rnn_module, f"bias_hh_l{l}").data)
        cells.append(cell)
    return cells


@torch.no_grad()
def _random_embeds(model, T=120, B=1, device="cpu"):
    ids = torch.randint(0, model.embed.num_embeddings, (B, T), device=device)
    return model.embed(ids)  # [B, T, E]


# ---- Step-by-step unroll of copied cells; return top-layer states: {"h": [...]}, plus "c" for LSTM ----
def unroll_cells_with_grad_seed(model, T=120, device="cpu"):
    model.eval()
    model.to(device)
    is_lstm = isinstance(model.rnn, nn.LSTM)

    cells = build_cells_from_module(model.rnn).to(device)
    x = _random_embeds(model, T=T, B=1, device=device)  # inputs don't need grads

    H, L = model.hidden_size, model.num_layers
    # Seed states with requires_grad=True so the recurrence is differentiable wrt h_0/c_0
    hs = [torch.zeros(1, H, device=device, requires_grad=True) for _ in range(L)]
    cs = [torch.zeros(1, H, device=device, requires_grad=True) for _ in range(L)]

    h_top_list, c_top_list = [], []
    for t in range(T):
        inp = x[:, t, :]  # [1, E]
        for l in range(L):
            if is_lstm:
                hs[l], cs[l] = cells[l](inp, (hs[l], cs[l]))
            else:
                hs[l] = cells[l](inp, hs[l])
            inp = hs[l]
        h_top_list.append(hs[-1])  # [1, H], requires_grad=True
        c_top_list.append(cs[-1])  # [1, H], requires_grad=True
    return {"h": h_top_list, "c": c_top_list} if is_lstm else {"h": h_top_list}


# ---- Jacobian-proxy norms for a sequence of states s_list (e.g., h_t or c_t) ----
def jacobian_proxy_norms(s_list, max_lag=60):
    """
    Proxy: phi(s_{t+s}) = 1/2 ||s_{t+s}||^2; grad wrt s_t gives (∂s_{t+s}/∂s_t)^T s_{t+s}.
    Returns lags [S], avg_norms [S], heat [T, S+1].
    """
    T = len(s_list)
    if T < 2:
        return torch.tensor([]), torch.tensor([]), torch.zeros(0, 0)
    S = min(max_lag, T - 1)
    device = s_list[0].device
    heat = torch.zeros(T, S + 1, device=device)
    lags, norms = [], []
    for s in range(1, S + 1):
        vals = []
        for t in range(0, T - s):
            phi = 0.5 * (s_list[t + s] ** 2).sum()
            (g_t,) = torch.autograd.grad(
                phi, (s_list[t],), retain_graph=True, allow_unused=True
            )
            if g_t is None:
                continue
            v = g_t.norm().item()
            vals.append(v)
            heat[t, s] = v
        if vals:
            lags.append(s)
            norms.append(sum(vals) / len(vals))
    return torch.tensor(lags), torch.tensor(norms), heat[:, : S + 1].detach().cpu()


def half_life(lags, norms, frac=0.5):
    if len(lags) == 0:
        return None
    base = norms[0].item()
    thr = frac * base
    for s, v in zip(lags.tolist(), norms.tolist()):
        if v <= thr:
            return s
    return None


def probe_grad(model, T=120, max_lag=60, device="cpu"):
    """Returns {state: (lags, norms, heat)} for top-layer h, plus c for LSTM."""
    states = unroll_cells_with_grad_seed(model, T=T, device=device)
    return {
        state: jacobian_proxy_norms(s_list, max_lag=max_lag)
        for state, s_list in states.items()
    }
