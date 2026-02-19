# sweet-watermark

\*\***updated (3/4/2024)**\*\* Our paper and repo are updated: DS-1000 benchmark is included, a new baseline (EXP-edit) is included for reproducing the main results. Experiments of using surrogate model, variable renaming, and detectability@T will be added.

## 环境部署 (Deployment)

### 快速部署 (一键脚本)

```bash
bash setup_env.sh
```

脚本会自动完成以下步骤。如需手动配置，请参考下面的分步说明。

### 分步部署

#### 1. 安装 Miniconda

如果尚未安装 Anaconda 或 Miniconda：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash
source ~/.bashrc
```

#### 2. 创建并激活 conda 环境

```bash
conda create -n sweet-watermark python=3.10 -y
conda activate sweet-watermark
```

#### 3. 安装 PyTorch (根据 CUDA 版本选择)

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 4. 安装项目依赖

```bash
pip install -r requirements.txt
```

#### 5. 系统依赖 (Cython 编译需要)

```bash
sudo apt install build-essential python3-dev   # Ubuntu/Debian
```

#### 6. 配置 accelerate

```bash
accelerate config    # 交互式配置，选择 GPU 数量、精度等
```

#### 7. HuggingFace 模型访问

使用 StarCoder 等受限模型前需登录并接受协议：

1. 在 https://huggingface.co/settings/tokens 创建 Access Token
2. 在 https://huggingface.co/bigcode/starcoder 接受模型使用协议
3. 设置环境变量 (推荐，添加到 `~/.bashrc` 永久生效):

```bash
export HF_TOKEN="hf_你的token"
```

或使用 Python 登录 (会将 token 缓存到本地):

```bash
python -c "from huggingface_hub import login; login(token='hf_你的token')"
```

## Introduction
Official repository of the paper:

"[Who Wrote this Code? Watermarking for Code Generation](https://arxiv.org/abs/2305.15060)" by [Taehyun Lee*](https://vision.snu.ac.kr/people/taehyunlee.html), [Seokhee Hong*](https://hongcheki.github.io/), [Jaewoo Ahn](https://ahnjaewoo.github.io/), [Ilgee Hong](https://ilgeehong.github.io/), [Hwaran Lee](https://hwaranlee.github.io/), [Sangdoo Yun](https://sangdooyun.github.io/), [Jamin Shin'](https://www.jayshin.xyz/), [Gunhee Kim'](https://vision.snu.ac.kr/gunhee/)

<p align="center">
    <img src="./img/main_table.png" alt="main table" width="80%" height="80%"> 
</p>
<p align="center">
    <img src="./img/pareto_figure.png" alt="Pareto Frontier" width="80%" height="80%"> 
</p>

## Reproducing the Main Experiments

### 1. Generating watermarked machine-generated code, calculating pass@k and detecting watermarks
We conducted our (main) experiments by separating them into generation and detection phases. However, anyone wanting to run both phases with a single command removes the `--generation_only` argument.

For EXP-edit with a high entropy setting, please set `top_p=1.0` and `temperature=1.0`.

### generation phase
```
bash scripts/main/run_{MODEL}_generation.sh
```

### detection phase
```
bash scripts/main/run_{MODEL}_detection.sh
```

### 2. Detecting watermarks in human-written code
```
bash scripts/main/run_{MODEL}_detection_human.sh
```

### 3. Calculating Metrics (AUROC, TPR)
With both metric output files from machine-generated and human-written codes, we calculate metrics including AUROC and TPR and update the results to `OUTPUT_DIRECTORY`.

```
python calculate_auroc_tpr.py \
    --task {humaneval,mbpp} \
    --human_fname OUTPUT_DIRECTORY_HUMAN \
    --machine_fname OUTPUT_DIRECTORY
```

## Acknowledgements
This repository is based on the codes of [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) in [BigCode Project](https://github.com/bigcode-project).

## Contact
If you have any questions about our codes, feel free to ask us: Taehyun Lee (taehyun.lee@vision.snu.ac.kr) or Seokhee Hong (seokhee.hong@vision.snu.ac.kr)
