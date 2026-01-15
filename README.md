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
# Setup
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

## Extract CLIP features
We first CLIP vision features from the dataset.
```text
python extract_clip_features.py
```
Output (UCF-Crime): 
```text
CLIP_feats
    └──ucf_test
          ├── Abuse028_x264_CLIP_features.npy
          ├── Abuse030_x264_CLIP_features.npy
          └── ...
    └──xd_test
          ├── A.Beautiful.Mind.2001__00-25-20_00-29-20_label_A_CLIP_features.npy
          ├── A.Beautiful.Mind.2001__00-40-52_00-42-01_label_A_CLIP_features.npy
          └── ...
```
# Inference
## 1. Generate VLM response
The VLM responses are obtained in JSON format.
```text
python generate_VLM_response_ucf.py
```
## 2. Local Response Cleaning (LRC)
The results of Local Response Cleaning (LRC) are saved in JSON format. We provide the generated responses in [`VLM_responses_LRC.json`](outputs/VLM_responses_LRC.json) for UCF-Crime and [`VLM_responses_LRC_xd.json`](outputs/LM_responses_LRC_xd.json) for XD-Violence.
```text
python LRC.py
```
## 3. Evaluation
Evaluation for UCF-Crime dataset
```text
eval_ucf.py
```
Evaluation for XD-Violence dataset
```text
eval_xd.py
```
# Acknowledgements
Thanks to [Ye et al.](https://github.com/vera-framework/VERA) for sharing their code.
