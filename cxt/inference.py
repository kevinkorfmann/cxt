import torch
from cxt.train import LitTokenFreeDecoder
import torch.nn.functional as F
from cxt.utils import generate_causal_mask


def generate_nokv(model, src, top_k=None, temperature=1, max_len=500, device='cuda'):
    with torch.no_grad():
        B = src.size(0)
        idx = torch.ones(B, 1).long().to(device)
        for _ in range(max_len):
            #with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(src, idx, calculate_loss=False)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx[:, 1:].cpu().numpy() - 2


def generate(model, src, B=20, device="cuda"):
    attn_mask = generate_causal_mask(1001, device)
    attn_mask = attn_mask.repeat(B, 1, 1, 1)
    idx = torch.ones(B, 1).long().to(device)
    with torch.no_grad():
        logits = model(src, None, attn_mask, calculate_loss=False, use_cache=True, position=0)
        idx = torch.ones(B, 1).long().to(device)
        top_k = 50
        for i in range(500, 1000):
                #with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits = model(src, idx[:, -1:], attn_mask, calculate_loss=False, use_cache=True, position=i)
                    logits = logits[:, -1, :]
                    if top_k is not None:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = -float('Inf')
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    idx = torch.cat([idx, next_token], dim=1)
    return idx

def load_model(config, model_path=None, device='cuda'):
    model = LitTokenFreeDecoder(config)
    model.load_state_dict(torch.load(model_path,
        weights_only=False)['state_dict'], strict=False)
    model = model.model
    model.to(device=device)
    model.eval()
    return model