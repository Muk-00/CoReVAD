import os, glob
import numpy as np
import torch
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoModel, AutoTokenizer
from internvl_utils import build_transform, dynamic_preprocess
import json
import torch.nn.functional as F
import math
from tqdm import tqdm

# 0) Load video path
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
            v = item["video"]
            v = v.replace("#", "")
            if not v.endswith(".mp4"):
                v = v + ".mp4"
            video_names.append(os.path.basename(v))

        for name in video_names:
            if name in all_basenames:
                selected_paths.append(all_basenames[name])
            else:
                print(f"[WARN] Not found in video_dir: {name}")

    return sorted(selected_paths)


# 1) Extract pixel_values
@torch.no_grad()
def get_pixel_values(vr, frame_indices, input_size=448, max_num=1):
    """
    return:
        pixel_values: [N, 3, 448, 448]
        num_patches_list: N
    """
    transform = build_transform(input_size=input_size)
    pixel_values_list, num_patches_list = [], []
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list, dim=0)

    return pixel_values, num_patches_list

# 2) Make VLM response
@torch.no_grad()
def make_VLM_response(model, tokenizer, generation_config, pixel_values, num_patches_list, device, Prompt_VLM):
    pixel_values = pixel_values.to(torch.bfloat16).to(device)
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    Prompt_VLM = Prompt_VLM.replace('$Data', video_prefix)
    response = model.chat(tokenizer, pixel_values, num_patches_list=num_patches_list, question=Prompt_VLM, 
                          generation_config=generation_config, return_history=False)

    return response

# 3) Divide frames from each segment (30-frame window, sample 8 frames)
def build_second_chunks(num_frames, divide_frames=30, num_per_sec=8):

    num_secs = math.ceil(num_frames / divide_frames)
    sec_frame_chunks = []
    for s in range(num_secs):
        start = s * divide_frames
        end = min((s + 1) * divide_frames, num_frames)
        if end - start <= num_per_sec:
            sampled = list(range(start, end))
        else:
            sampled = np.linspace(start, end - 1, num=num_per_sec, dtype=np.int32).tolist()
        sec_frame_chunks.append(sampled)
    
    return sec_frame_chunks


def test(selected_videos_test, model, tokenizer, generation_config, Prompt_VLM, device, save_name="VLM_responses"):
    for video_path in tqdm(sorted(selected_videos_test), desc="Processing videos", ncols=90):
        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            num_frames = len(vr)
            fname_no_ext = os.path.splitext(os.path.basename(video_path))[0]
            divide_frames = 30
            sec_frame_chunks = build_second_chunks(num_frames, divide_frames=divide_frames, num_per_sec=8)
            selected_texts = []

            for frame_chunk  in sec_frame_chunks:
                pixel_values_q, num_patches_q = get_pixel_values(vr, frame_chunk)
                response__learner_q = make_VLM_response(model, tokenizer, generation_config, pixel_values_q, num_patches_q, device, Prompt_VLM)
                text = response__learner_q.strip().split("\n")
                selected_text = text[1].strip() +" "+ text[2].strip()
                selected_texts.append(selected_text)
            # 2) saved text
            save_path = f"./outputs/{save_name}.json"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if os.path.exists(save_path):
                with open(save_path, "r", encoding="utf-8") as f:
                    all_texts = json.load(f)
            else:
                all_texts = {}

            if fname_no_ext in all_texts:
                all_texts[fname_no_ext].extend(selected_texts)
            else:
                all_texts[fname_no_ext] = selected_texts

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(all_texts, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] Eval failed on {video_path}: {e}")
            continue

if __name__ == "__main__":
    save_name="VLM_responses"
    mllm_path = 'OpenGVLab/InternVL2-8B'
    device = torch.device('cuda:0')
    video_dir = "/path/to/UCF-Crime/videos/"
    test_path = "/path/to/UCF-Crime/Anomaly_Detection_splits/Anomaly_Test.txt"

    with open('P_VLM_format.txt', 'r', encoding='utf-8') as f:
        Prompt_VLM = f.read()

    model = AutoModel.from_pretrained(mllm_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(mllm_path, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=1024, do_sample=False)

    selected_videos_test = load_selected_video_paths(video_dir, test_path)

    test(selected_videos_test, model, tokenizer, generation_config, Prompt_VLM, device, save_name=save_name)