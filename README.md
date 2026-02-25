# Multi-Task Diffusion-Adapter Model for MRI Synthesis Across Multiple Field Strengths



**Abstract**  <br />
Magnetic resonance imaging across different field strengths presents distinct trade-offs in cost, accessibility and diagnostic quality. For instance, low-field scanners provides portable and cost-effective method but produce images with a lower signal-to-noise ratio, while ultra-high-field scanners give higher resolution imaging yet remain limited in access. Deep learning-based synthesis can bridge these quality gaps, but existing methods address each field-strength transition independently, as multiple synthesis directions within a single framework are challenging due to different degradation characteristics, requiring distinct recovery strategies across the enhancement tasks. To address this, we propose an adaptive multi-task diffusion approach that supports multiple cross-field syntheses while adapting to their specific requirements through task-specific adaptation. As each tasks demand different forms of structural recovery at each denoising step, we introduce context-aware feature modulation conditioned on the noise level to separately regulate coarse reconstruction and fine-detail refinement during the synthesis. We also introduce region-specific anatomical constraints to further ensure the structural fidelity of the synthetic results. Experiments on paired 64mT/3T/7T brain MRI demonstrate accurate synthesis across field-strength transitions, with downstream task performance further validating the effective enhancement toward higher-field imaging quality.

**Architecture**  <br />

![alt text](figures/Figure1.jpg)

**System Requirement**  <br />
All the experiments are conducted on Ubuntu 20.04 Focal version with Python 3.8.

To train the model with the given settings, the system requires a GPU with at least 40GB. All the experiments are conducted on two Nvidia A40 GPUs. (Not required any non-standard hardware)

***Installation Guide***  <br />
Prepare an environment with python>=3.8 and install dependencies
```
pip install -r requirements.txt
```

**Dataset Preparation**  <br />
The experiments are conducted on a in-house paired 64mT-3T and 3T-7T dataset,
  * UNC 3T-7T Dataset : [https://springernature.figshare.com/articles/dataset/UNC_Paired_3T-7T_Dataset/23706033](https://springernature.figshare.com/articles/dataset/UNC_Paired_3T-7T_Dataset/23706033)
  * 
Separate each field MRI .npy slices using pre_process.py. 
```
python pre_process.py
```
 Then, save the .npy data as in the following structure.
```
data/
├── T1/
│   ├── train/
│   │   ├── lf_t1.npy
│   │   └── hf_t1.npy
│   │   └── 3T_t1.npy
│   │   └── 7T_t1.npy
│   ├── test/
│   │   ├── lf_t1.npy
│   │   └── hf_t1.npy
│   │   └── 3T_t1.npy
│   │   └── 7T_t1.npy
│   ├── val/
│   │   ├── lf_t1.npy
│   │   └── hf_t1.npy
│   │   └── 3T_t1.npy
│   │   └── 7T_t1.npy
```


**Train Model**  <br />
To train the model on the multi-field datasets.
```
python train.py --image_size 256 --exp exp_multiField --num_channels 1 --num_channels_dae 64 --ch_mult 1 2 4 --num_timesteps 4 --num_res_blocks 2 --batch_size 3 --num_epoch 30 --ngf 64 --embedding_type positional --ema_decay 0.999 --r1_gamma 1. --z_emb_dim 256 --lr_d 1e-4 --lr_g 1.6e-4 --lazy_reg 10 --num_process_per_node 3
```


**Test Model**  <br />
```
python test.py --image_size 256 --exp exp_multiField  --num_channels 1 --num_channels_dae 64 --ch_mult 1 2 4 --num_timesteps 4 --num_res_blocks 2 --batch_size 1 --embedding_type positional  --z_emb_dim 256  --gpu_chose 0 --input_path '/data/T1' --output_path '/results'
```

**Acknowledgements**  <br />
This repository makes liberal use of code from [Tackling the Generative Learning Trilemma](https://github.com/NVlabs/denoising-diffusion-gan) and [SynDiff](https://github.com/icon-lab/SynDiff)
