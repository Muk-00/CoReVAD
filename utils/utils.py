import numpy as np
import torch
import torch.nn.functional as F
import clip

@torch.no_grad()
def text_encode(response: str, clip_model, device):
    if not response:
        return torch.zeros(512, device=device)
    tokens = clip.tokenize(response, truncate=True).to(device)
    text_feat = clip_model.encode_text(tokens).float()
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    return text_feat.squeeze(0)

@torch.no_grad()
def cosine_sim_text_vs_frames(img_feats: np.ndarray, text_feat: torch.Tensor, device):
    img = torch.from_numpy(img_feats).to(device).float()

    return ((img @ text_feat)).cpu().numpy()
