# 기본 refine함수 없이 0,1로 AUROC 계산

import os, json, re, argparse
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import clip
from base_cos_sim import text_encode, cosine_sim_text_vs_frames
from math import ceil
from decord import VideoReader, cpu
from tqdm import tqdm

import pandas as pd
# ==========================
# 2) 첫 문장 → 초기 a_k (0/1) & 본문 텍스트
# ==========================
def load_selected_texts_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)  # {video: [seg0_text, seg1_text, ...]}

def parse_float_list(s):
    return [float(x) for x in s.split(',') if x.strip()]

def parse_int_list(s):
    return [int(float(x)) for x in s.split(',') if x.strip()]

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

# ==========================
# 4) Gaussian smoothing & Position weighting
# ==========================
def gaussian_kernel_original(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) 

def gaussian_smoothing(data, sigma):
    n = len(data)
    smoothed_data = np.zeros(n)
    x = np.arange(n)

    centroid_index = int(n/2)
    # centroid_index = np.argmax(data)

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


# ==========================
# 4) 비디오 처리 → 프레임 스코어 (seg_num 균등 분할 + 프레임 길이 슬라이스)
# ==========================
@torch.no_grad()
def process_video(video, texts, clip_feature_root_test, vr_len, clip_model, device,
                    K=10, tau=1.0, alpha=0.33, size=29, sigma=10, edge_len=30, frame_len=30):

    """
    - CLIP feature: {video}_CLIP_features.npy  (이미 16프레임 간격으로 샘플링)
    - 문장 1개가 seg_num 샘플 프레임을 커버한다고 가정 → T를 seg_num로 균등 분할
    - text_encode, cosine_sim_text_vs_frames 사용
    - 반환: y_pred (길이=vr_len)
    """
    npy_path = os.path.join(clip_feature_root_test, f"{video}_CLIP_features.npy")
    clip_feats = np.load(npy_path).astype(np.float16)  # (T, D)
    T = clip_feats.shape[0]
    M = len(texts)

    # video features 임베딩
    segments = []
    seg_feats_list = []
    for i in range(0, T, frame_len):
        start = i
        end   = min(i + frame_len, T)  # 마지막은 남은 프레임만
        segments.append((start, end))
    M_seg = len(segments)
    for (s, e) in segments:
        seg_feats_list.append(clip_feats[s:e].mean(axis=0, keepdims=True))  # (1, D)
    seg_feats = np.concatenate(seg_feats_list, axis=0).astype(np.float16)  # (M, D)

    # 1) 초기 a_k, 텍스트 임베딩
    y_init = []
    text_emb_list = []
    for txt in texts:
        first, body = split_first_sentence(txt)
        y_init.append(first_sentence_to_label(first))          # 1/0
        t_emb = text_encode(body, clip_model, device)          # (1,D), L2 norm 가정
        text_emb_list.append(t_emb)
    y_init = np.asarray(y_init, dtype=np.float16)

    # cosine similarity
    sims = np.zeros((M_seg, M), dtype=np.float16)
    for k in range(M):
        sims_full = cosine_sim_text_vs_frames(seg_feats, text_emb_list[k], device)  # (T,)
        if torch.is_tensor(sims_full):
            sims[:,k] = sims_full.detach().cpu().float().numpy()
        else:
            sims[:,k] = sims_full.astype(np.float16)

    # ======= ✅ top-k 대신 전체 텍스트에 대해 softmax 가중합 =======
    # 각 segment t에 대해 모든 문장 k에 대해 softmax(sims[t, k]/tau)로 weight 계산
    sims_centered = sims - sims.max(axis=1, keepdims=True)           # 수치 안정화
    exps = np.exp(sims_centered / float(tau)).astype(np.float16)     # (M_seg, M)
    weights = exps / (exps.sum(axis=1, keepdims=True) + 1e-12)       # (M_seg, M)
    y_refined = (weights * y_init[None, :]).sum(axis=1).astype(np.float16)  # (M_seg,)

    # gaussian smoothing
    y_pred = gaussian_smooth_1d(np.array(y_refined), size, sigma) # Gaussian smoothing (1)
    # flatten
    y_pred = np.repeat(y_pred, frame_len)[:vr_len].astype(np.float16)
    # Posisition weighting
    sigma1 = int(len(y_pred)*0.5)
    y_pred = gaussian_smoothing(y_pred, sigma1)    # Gaussian smoothing (2)

    return y_pred

# ==========================
# 5) 메인: 전체 비디오 concat + GT npy로 AUROC
# ==========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window_size",    type=int, default=2)
    ap.add_argument("--clip_feature_root_test", default="./clip_feats_all/ucf_test")
    ap.add_argument("--videos_dir",  default="/home/lhm/UCF-Crimes/videos")  # 비디오 디렉토리
    ap.add_argument("--test_list",   default="/home/lhm/UCF-Crimes/Anomaly_Detection_splits/Anomaly_Test.txt")  # sum_anomaly_scores7.py와 동일 순서 파일
    ap.add_argument("--gt_npy",      default="./gt_ucf_testorder.npy")  # 프레임 단위 GT (npy)
    ap.add_argument("--device",      default="cuda")

    ap.add_argument("--tau_list", type=str, default="0.01,0.05,0.1,0.2")
    ap.add_argument("--size_list", type=str, default="5,9,13,15,17,19,21,25,29,31")
    ap.add_argument("--sigma_list", type=str, default="1,5,10,15,20")
    
    ap.add_argument("--frame_len",    type=int, default=30)
    args = ap.parse_args()

    # args.texts_json  = f"./selected_texts_4v_10_no_q.txt" # basellne
    # args.texts_json  = f"./selected_texts_4v_10_no_q_all_body.json" # LAVAD
    args.texts_json  = f"./selected_texts_4v_10_no_q_{args.window_size}.json" # Ours
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    clip_model, _ = clip.load("ViT-B/16", device=device) # ViT-B/16
    clip_model.eval()

    # 데이터 로드
    # print(f'vid2segs keys: {list(vid2segs.keys())[:5]}')
    vid2texts = load_selected_texts_json(args.texts_json)

     # --- K/tau/alpha/sigma 리스트 파싱 ---
    # alpha_list = parse_float_list(args.alpha_list)
    tau_list   = parse_float_list(args.tau_list)
    size_list = parse_int_list(args.size_list)
    sigma_list = parse_int_list(args.sigma_list)

    tau = tau_list[0]
    # edge_len_list = parse_int_list(args.edge_len)

    print(f"[INFO] tau_list: {tau_list} | size_list: {size_list} | sigma_list: {sigma_list}")

    with open(args.test_list, 'r', encoding='utf-8') as f:
        test_names = [os.path.basename(x.strip()).replace(".mp4","") for x in f]

    y_true = np.load(args.gt_npy).astype(np.int8)

    results = []  # 모든 실험 결과를 저장할 리스트
    out_json_path = f"ablation_study_LRC.json"
    print(f'out_json_path: {out_json_path}')
    
    score_root = "./score_ucf"  # save UCF-Crime AUC
    os.makedirs(score_root, exist_ok=True)

    for size in size_list:
        for sigma in sigma_list:
            preds = []
            desc = f"K={1.0}, tau={tau}, size={size}"
            for vid in tqdm(test_names, desc=desc, total=len(test_names), leave=False):
                vpath = os.path.join(args.videos_dir, vid + ".mp4")
                vr = VideoReader(vpath, ctx=cpu(0), num_threads=1)
                texts = vid2texts[vid]

                y_pred_frames = process_video(
                    video=vid,
                    texts=texts,
                    clip_feature_root_test=args.clip_feature_root_test,
                    clip_model=clip_model,
                    device=device,
                    vr_len=len(vr),
                    K=1.0,
                    tau=tau,
                    alpha=0.11,
                    size=size,
                    sigma=sigma,
                    edge_len=1,
                    frame_len=args.frame_len
                )
                # Save AUC
                save_path = os.path.join(score_root, f"{vid}_scores.npy")
                np.save(save_path, y_pred_frames.astype(np.float16))
                
                preds.append(y_pred_frames)

            y_score = np.concatenate(preds, axis=0)
            auroc = roc_auc_score(y_true, y_score)
            ap1   = average_precision_score(y_true, y_score)

            print(f"[UCF-Crimes RESULT] tau={tau}, size={size} sigma={sigma} "
                f"=> AUROC={auroc*100:.2f}%, AP={ap1*100:.2f}%")

            # JSON에 저장할 항목
            results_item = {
                "texts_json" : args.texts_json,
                "window_size" : args.window_size,
                "tau": tau,
                "size" : size,
                "sigma" : sigma,
                "AUROC": float(auroc),
            }
            results.append(results_item)
            # === JSON으로 저장 ===
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"[INFO] Grid search results saved to {out_json_path}")

if __name__ == "__main__":
    main()