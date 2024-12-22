import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from cxt.modules import MutationsToLatentSpace
from cxt.modules import Block
from cxt.modules import LayerNorm



class TokenFreeDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            bt2ls = MutationsToLatentSpace(config=config),
            ote = nn.Embedding(config.output_dim, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, use_bias=config.bias),))
        self.lm_head = nn.Linear(config.n_embd, config.output_dim, bias=False)
        self.cached_src = None
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, x, y, attn_mask, position=None, use_cache=False, calculate_loss=True):
        if use_cache:
            pass
        else:
            B, _ = y.size()
            B, _, _, NW, _ = x.size()

             # src and tgt will be created on the fly 
            x = self.transformer.bt2ls(x)   # -> [B, x_len, E]
            y2 = self.transformer.ote(y)     # -> [B, y_len, E]
            #print(f"X: {x.size()} Y: {y.size()}")
            src = torch.cat([x, y2], dim=1)  # -> [B, L, E]
            src = self.transformer.drop(src)

            if calculate_loss:
                zero_col = torch.zeros(B, 1, device=y.device)
                # removing start token
                tgt = torch.cat((y[:, 1:], zero_col), dim=1)

            for block in self.transformer.h:
                src = block(src, attn_mask, use_cache=False)
            src = self.transformer.ln_f(src)

            logits = self.lm_head(src)
            logits = logits[:, NW:, :].contiguous()
            
            if calculate_loss:
                tgt = tgt.long()
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                        tgt.reshape(-1))
                return logits, loss
            else: return logits
