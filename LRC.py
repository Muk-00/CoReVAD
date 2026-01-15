import json, os
import numpy as np
import torch
from tqdm import tqdm
from utils import text_encode, cosine_sim_text_vs_frames
import clip

@torch.no_grad()
def Local_Response_Cleaning(clip_feat_dir, device, window_size=1, frame_len=30):
    
    response_path = f"./outputs/VLM_responses.json"
    LRC_path = f"./outputs/VLM_responses_LRC.json"

    # 0) Load
    with open(response_path, "r", encoding="utf-8") as f:
        response = json.load(f)

    # 1) Load CLIP
    clip_model, _ = clip.load("ViT-B/16", device=device, jit=False)
    clip_model.eval()

    replaced = {}
    for vid, responses in tqdm(response.items(), desc="Videos", total=len(response)):
        feat_path = os.path.join(clip_feat_dir, f"{vid}_CLIP_features.npy")

        img_feats = np.load(feat_path).astype(np.float32)
        v, _ = img_feats.shape
        img_feats_list = []
        for i in range(0, v, frame_len):
            start = i
            end = min(i + frame_len, v)
            img_feats_list.append(img_feats[start:end].mean(axis=0, keepdims=True))

        vision_feats = np.concatenate(img_feats_list, axis=0).astype(np.float32)
        v_seg = vision_feats.shape[0]

        responses = [" ".join(r.splitlines()).strip() if r else "" for r in responses]
        text_feats = []
        for r_tilde in responses:
            t = text_encode(r_tilde, clip_model, device)  # (512,)
            text_feats.append(t)
        text_feats = torch.stack(text_feats, dim=0)  # (M, 512)
        M=text_feats.shape[0]

        sims_TM = []
        for j in range(len(responses)):
            sims_j = cosine_sim_text_vs_frames(vision_feats, text_feats[j], device)  # (v_seg,)
            sims_TM.append(sims_j)
        sims_TM = np.stack(sims_TM, axis=1)

        # Local Response Cleaning
        new_responses = []
        for i in range(v_seg):
            j_min = max(0, i - window_size)
            j_max = min(M, i + window_size + 1)
            local_candidates = np.arange(j_min, j_max)
            local_scores = sims_TM[i, local_candidates]
            j_star = local_candidates[np.argmax(local_scores)]
            r_bar = responses[j_star]
            new_responses.append(r_bar)

        replaced[vid] = new_responses

    # 5) Save
    os.makedirs(os.path.dirname(LRC_path) or ".", exist_ok=True)
    with open(LRC_path, "w", encoding="utf-8") as f:
        json.dump(replaced, f, ensure_ascii=False, indent=2)
    print(f"[OK] saved → {LRC_path}")


if __name__ == "__main__":
    clip_feat_dir = "./CLIP_feats/ucf_test"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Local_Response_Cleaning(clip_feat_dir=clip_feat_dir, device=device)