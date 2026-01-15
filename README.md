# CoReVAD: A-Contextual-Reasoning-Framework-for-Training-Free-Video-Anomaly-Detection

<p align="center">
  <img src="image/main_framework.png" width="1000">
</p>

> <b>Abstract: </b>Existing Video Anomaly Detection (VAD) methods typically rely on task-specific training, leading to strong domain dependency and high training costs.
> Moreover, most existing methods output only scalar anomaly scores, providing limited insight into why specific events are considered abnormal.Recent advances in Vision–
> Language Models (VLMs) have enabled both anomaly detection and human-interpretable reasoning. However, many VLM-based approaches still require additional training steps
> (e.g., instruction tuning or verbalized learning) or external Large Language Models (LLMs), incurring further training costs and inference overhead. To address these
> challenges, we propose CoReVAD, a contextual reasoning framework for training-free video anomaly detection that operates with a single frozen VLM. CoReVAD directly generates
> anomaly scores and temporal descriptions from the VLM. To mitigate noise in generative outputs, we introduce a Local Response Cleaning (LRC) module based on local vision–
> text alignment. Furthermore, global temporal context and progression are incorporated through softmax-based refinement, Gaussian smoothing, and position weighting.
> Experiments on UCF-Crime and XD-Violence demonstrate that CoReVAD achieves competitive performance among training-free methods while providing reliable and interpretable
> explanations.

# Data
For datasets, Please download the original videos from links (GT of each datasets is already included).
- UCF-Crime: [link](https://www.crcv.ucf.edu/projects/real-world/)
- XD-Violence: [link](https://roc-ng.github.io/XD-Violence/)

The test video directory structure is as follows:
```text
UCF-Crime
    └── videos
          ├── Abuse028_x264.mp4
          ├── Abuse030_x264.mp4
          └── ...
XD-Violence
    └── videos
          ├── A.Beautiful.Mind.2001__00-25-20_00-29-20_label_A.mp4
          ├── A.Beautiful.Mind.2001__00-40-52_00-42-01_label_A.mp4
          └── ...
```
# Install
## Clone the repo
```text
git clone https://github.com/Muk-00/CoReVAD.git
cd CoReVAD
conda create --name CoReVAD python=3.9
conda activate CoReVAD
pip install -r requirements.txt
```
## Install the environment
In this paper, we use InternVL2, we follow the official installation instructions provided by InternVL2 ([link](https://internvl.readthedocs.io/en/latest/get_started/installation.html))

# Inference
## 1. Extract CLIP features
실험하기에 앞서 우선, 데이터셋의 CLIP vision feature를 추출해야한다.
```text
python extract_clip_features.py
```
Output (UCF-Crime): 
```text
CLIP_feats
    └──ucf_text
          ├── Abuse028_x264_CLIP_features.npy
          ├── Abuse030_x264_CLIP_features.npy
          └── ...
```
## 2. VLM response 출력
VLM 출력 결과를 json file을 통해 얻을 수 있으며, 우리는 해당 결과인 VLM_responses.json을 제공한다.
```text
python generate_VLM_response.py
```
## 3. Local Response Cleaning (LRC)
LRC 결과를 json file을 통해 얻을 수 있으며, 우리는 해당 결과인 VLM_responses_LRC.json을 제공한다.
```text
python LRC.py
```
## 4. Evaluation
```text
eval.py
```
