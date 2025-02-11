import torch
from cxt.train import LitTokenFreeDecoder
import torch.nn.functional as F
from cxt.utils import generate_causal_mask
from itertools import combinations
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np

from cxt.utils import post_process, accumulating_mses, mse
from cxt.utils import simulate_parameterized_tree_sequence, TIMES


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


def generate(model, src, B=20, device="cuda", clear_cache=True):
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
    
    if clear_cache: model.clear_cache()
    return idx

def load_model(config, model_path=None, device='cuda'):
    model = LitTokenFreeDecoder(config)
    model.load_state_dict(torch.load(model_path,
        weights_only=False)['state_dict'], strict=False)
    model = model.model
    model.to(device=device)
    model.cache_to_device(device)
    model.eval()
    return model

def prepare_ts_data(ts: object, num_samples: int, B: int, device='cuda') -> tuple:
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
    src = src.to(device).to(torch.float32)
    return src, tgt

def translate_from_ts(
        ts,
        max_replicates = 20, use_early_stopping = True,
        model_config=None,
        model_path=None,
        device='cuda'
    ):
    """Assumes sample size of 50 for now and large enough GPU to fit 1225 batch."""

    model = load_model(
        config=model_config, 
        model_path=model_path,
        device=device
    )

    src, tgt = prepare_ts_data(ts, num_samples=50, B=1225, device=device)
    yhats, ytrues = [], []
    for i in range(max_replicates):
        sequence = generate(model, src, B=1225, device=device)
        yhat, ytrue = post_process(tgt, sequence, TIMES)
        yhats.append(yhat)
        ytrues.append(ytrue)
        if use_early_stopping:
            # early stopping criteria
            if i > 1:
                mses = accumulating_mses(yhats, ytrues)
                derivatives = np.diff(mses)
                if abs(derivatives[-1]) < 0.001:
                    print(f"Stopping at {i} because derivative is {derivatives[-1]}.")
                    break
    yhats = np.stack(yhats)
    ytrues = np.stack(ytrues)  
    return yhats, ytrues


