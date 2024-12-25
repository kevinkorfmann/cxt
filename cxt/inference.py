import torch

def generate(model, src, top_k=None, temperature=1, max_len=500, device='cuda'):
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