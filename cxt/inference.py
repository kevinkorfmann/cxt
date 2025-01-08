import torch
from cxt.train import LitTokenFreeDecoder
import torch.nn.functional as F
from cxt.utils import generate_causal_mask
from itertools import combinations
from multiprocessing import Pool
from tqdm import tqdm

from cxt.utils import process_pair
def totensorlist(l): return [torch.tensor(a) for a in l]


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

def prepare_ts_data(ts: object, num_samples: int, B: int) -> tuple:
    """
    Prepares the data for the model by processing pairs of samples from the tree sequence.

    Parameters:
    - ts: The tree sequence object.
    - num_samples: The number of samples to generate.
    - B: The batch size.

    Returns:
    - src: Tensor containing the source data.
    - tgt: Tensor containing the target data.
    """
    args = [(ts, a, b) for a, b in combinations(range(num_samples), 2)]
    with Pool(50) as pool: 
        results = list(tqdm(pool.imap(process_pair, args), total=B))
    src_list, tgt_list = zip(*results)
    src_list = totensorlist(src_list)
    tgt_list = totensorlist(tgt_list)
    src = torch.stack(src_list, dim=0)  
    tgt = torch.stack(tgt_list, dim=0) 
    return src, tgt