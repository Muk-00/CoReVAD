import os, json, re, argparse
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import clip
from utils import text_encode, cosine_sim_text_vs_frames
from decord import VideoReader, cpu
from tqdm import tqdm

# =========== utils =============
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def split_first_sentence(text: str):
    t = re.sub(r'\s+', ' ', text).strip()
    m = re.search(r'\.(\s|$)', t)
    if m:
        first = t[:m.end()].strip()
        body  = t[m.end():].strip()
    else:
        first, body = t, ""
    if not body:
        body = first
    return first, body

def first_sentence_to_label(first_sentence: str) -> float:
    score_init = first_sentence.lower()
    score = 0.0
    if score_init.startswith("anomalous scenes"):
        score = 1.0
    if score_init.startswith("normal scenes"):
        score = 0.0
    return score

def gaussian_kernel_original(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) 

def gaussian_smoothing(data, sigma):
    n = len(data)
    smoothed_data = np.zeros(n)
    x = np.arange(n)

    centroid_index = int(n/2)

    kernel_values = gaussian_kernel_original(x, centroid_index, sigma)
    smoothed_data =  kernel_values * data  

    return smoothed_data

def gaussian_kernel(size, sigma):
    kernel = np.exp(-np.linspace(-size//2, size//2, size)**2 / (2*sigma**2))
    return kernel / kernel.sum() 

def gaussian_smooth_1d(data, size=5, sigma=1.0):
    kernel = gaussian_kernel(size, sigma)
    smoothed_data = np.convolve(data, kernel, mode='same')
    return smoothed_data

# =========== eval =============
def process_video(video, clean_response, clip_feature_root_test, vr_len, clip_model, device, tau=0.05, size=9, sigma=5, frame_len=30):

    npy_path = os.path.join(clip_feature_root_test, f"{video}_CLIP_features.npy")
    clip_feats = np.load(npy_path).astype(np.float32)  # (T, D)
    T = clip_feats.shape[0]
    R_length = len(clean_response)

    vision_feats_list = []
    for i in range(0, T, frame_len):
        start = i
        end   = min(i + frame_len, T)
        vision_feats_list.append(clip_feats[start:end].mean(axis=0, keepdims=True))  # (1, D)

    vision_feats = np.concatenate(vision_feats_list, axis=0).astype(np.float16)  # (M, D)
    y_init = []
    response_emb_list = []
    for text in clean_response:
        first, body = split_first_sentence(text)
        y_init.append(first_sentence_to_label(first))
        r_emb = text_encode(body, clip_model, device)          
        response_emb_list.append(r_emb)
    y_init = np.asarray(y_init, dtype=np.float16)

    # cosine similarity
    sims = np.zeros((len(vision_feats_list), R_length), dtype=np.float16)
    for k in range(R_length):
        sims_full = cosine_sim_text_vs_frames(vision_feats, response_emb_list[k], device)  # (T,)
        if torch.is_tensor(sims_full):
            sims[:,k] = sims_full.detach().cpu().float().numpy()
        else:
            sims[:,k] = sims_full.astype(np.float16)

    sims_centered = sims - sims.max(axis=1, keepdims=True)
    exps = np.exp(sims_centered / float(tau)).astype(np.float16) 
    weights = exps / (exps.sum(axis=1, keepdims=True) + 1e-12)      
    y_refined = (weights * y_init[None, :]).sum(axis=1).astype(np.float16)

    # Gaussian smoothing
    y_pred = gaussian_smooth_1d(np.array(y_refined), size, sigma)
    # Flatten
    y_pred = np.repeat(y_pred, frame_len)[:vr_len].astype(np.float16)
    # Position weighting
    sigma1 = int(len(y_pred)*0.5)
    y_pred = gaussian_smoothing(y_pred, sigma1)

    return y_pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_feature_root_test", default="./CLIP_feats/xd_test")
    ap.add_argument("--videos_dir",  default="/path/to/XD-Violence/test/")
    ap.add_argument("--test_list",   default="/path/to/XD-Violence/xd_annotations.json")
    ap.add_argument("--gt_npy",      default="./gt_xd.npy")
    ap.add_argument("--device",      default="cuda")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--size", type=int, default=9)
    ap.add_argument("--sigma", type=int, default=5)
    ap.add_argument("--frame_len",    type=int, default=30)
    args = ap.parse_args()

    args.texts_json = f"./outputs/VLM_responses_LRC_xd.json"
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    clip_model, _ = clip.load("ViT-B/16", device=device) # ViT-B/16
    clip_model.eval()

    vid2texts = load_json(args.texts_json)

    with open(args.test_list, 'r', encoding='utf-8') as f:
        anno_list = json.load(f)

    vid_names = []
    vid_paths = []

    for item in anno_list:
        v_raw = item["video"]
        v_clean = v_raw.replace("#", "")
        if not v_clean.endswith(".mp4"):
            v_clean = v_clean + ".mp4"
        base = os.path.splitext(os.path.basename(v_clean))[0]
        vpath = os.path.join(args.videos_dir, os.path.basename(v_clean))
        vid_names.append(base)
        vid_paths.append(vpath)

    y_true = np.load(args.gt_npy).astype(np.int8)
    preds = []
    for vid, vpath in tqdm(zip(vid_names, vid_paths), desc="Processing videos", total=len(vid_names)):
        vr = VideoReader(vpath, ctx=cpu(0), num_threads=1)
        clean_response = vid2texts[vid]
        y_pred_frames = process_video(video=vid, clean_response=clean_response, clip_feature_root_test=args.clip_feature_root_test, clip_model=clip_model,
                                       device=device, vr_len=len(vr), tau=args.tau, size=args.size, sigma=args.sigma, frame_len=args.frame_len)
        
        preds.append(y_pred_frames)

    y_score = np.concatenate(preds, axis=0)
    auroc = roc_auc_score(y_true, y_score)
    ap1 = average_precision_score(y_true, y_score)
    print(f"[XD-Violence RESULT] => AUROC={auroc*100:.2f}%, AP={ap1*100:.2f}%")

if __name__ == "__main__":
    main()