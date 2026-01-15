# CLIP 비전 특징 추출기

import torch
import torch.nn.functional as F
from clip import clip
from internvl_utils import build_transform, dynamic_preprocess
from decord import VideoReader, cpu
import os, glob
from PIL import Image
import numpy as np

def get_pixel_values(vr, frame_indices, input_size=224, max_num=1):
    transform = build_transform(input_size=input_size)
    pixel_values_list, num_patches_list = [], []
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)  # (N, 3, 448, 448)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)

    return pixel_values, num_patches_list

def load_clip(device: str = "cuda", model_name: str = "ViT-B/16"):
    model, _ = clip.load(model_name, device=device)
    for p in model.parameters():
        p.requires_grad = False   
    model.eval()
    
    return model

@torch.no_grad()
def get_clip_frame_features(vr, frame_indices, model, batch_size = 16, input_size= 224, max_num = 1):
    # 1) extract pixel_values
    pixel_values, num_patches_list = get_pixel_values(vr, frame_indices, input_size=input_size, max_num=max_num)
    model_dtype = model.visual.conv1.weight.dtype
    pixel_values = pixel_values.to(device=device, dtype=model_dtype)

    # 2) extract tile embedding
    tiles = []
    total = pixel_values.shape[0]
    for s in range(0, total, batch_size):
        e = min(s + batch_size, total)
        feats = model.encode_image(pixel_values[s:e])
        feats = F.normalize(feats, dim=-1)
        tiles.append(feats)
    tile_features = torch.cat(tiles, dim=0)

    # 3) frame embedding by averaging tile embeddings
    frame_features, cur = [], 0
    for n in num_patches_list:
        if n > 0:
            f = tile_features[cur:cur+n].mean(dim=0)
        else:
            f = torch.zeros(tile_features.shape[1], device=device, dtype=tile_features.dtype)
        frame_features.append(f)
        cur += n
    frame_features = torch.stack(frame_features, dim=0)

    return frame_features

def load_selected_video_paths(video_dir, test_path):
    video_dir_lower = video_dir.lower()

    all_videos = sorted(glob.glob(os.path.join(video_dir, '*.mp4')))
    all_basenames = {os.path.basename(v): v for v in all_videos}
    selected_paths = []
    
    if "ucf" in video_dir_lower:
        with open(test_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        video_names = [os.path.basename(line.strip()) for line in lines]        
        
        for name in video_names:
            if name in all_basenames:
                selected_paths.append(all_basenames[name])
            else:
                print(f"[WARN] Not found in video_dir: {name}")
    
    else: # xd
        with open(test_path, "r", encoding="utf-8") as f:
            anno_list = json.load(f)
        
        video_names = []
        for item in anno_list:
            v = item["video"]  # 예: "A.Beautiful.Mind.2001__#01-14-30_01-16-59_label_A"
            v = v.replace("#", "")  # 실제 파일명에는 # 없음
            if not v.endswith(".mp4"):
                v = v + ".mp4"
            video_names.append(os.path.basename(v))

        for name in video_names:
            if name in all_basenames:
                selected_paths.append(all_basenames[name])
            else:
                print(f"[WARN] Not found in video_dir: {name}")

    return sorted(selected_paths)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    video_dir = "/path/to/UCF-Crime/videos/"
    annotation_path = "/path/to/UCF-Crime/Anomaly_Detection_splits/Anomaly_Test.txt"

    clip_model = load_clip(device="cuda", model_name="ViT-B/16")
    selected_videos = load_selected_video_paths(video_dir, annotation_path)
    save_dir = "./CLIP_feats/ucf_test"
    os.makedirs(save_dir, exist_ok=True)
    
    for idx, video_path in enumerate(sorted(selected_videos), start=1):
        fname = os.path.basename(video_path)
        fname_no_ext = os.path.splitext(fname)[0]  # 'Abuse028_x264'
        print(f"[INFO] Processing {idx} / {len(selected_videos)} videos")
        
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        all_frame_indices = list(range(len(vr)))
        clip_features = get_clip_frame_features(vr, all_frame_indices, clip_model, batch_size = 8, input_size = 224, max_num = 1)
        save_path = os.path.join(save_dir, f"{fname_no_ext}_CLIP_features.npy")
        np.save(save_path, clip_features.cpu().numpy())
        print(f"[INFO] Saved clip features to {save_path}")