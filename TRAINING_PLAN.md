# Brain-to-Text '25 Competition — Colab-First Training Plan (with Kaggle fallback)

Goal
- Produce a valid competition submission (CSV with id,text) using this repo’s baseline pipeline: GRU phoneme decoder + n‑gram LM (+ optional OPT rescoring).

Key facts from the repo
- Training code lives in `model_training/` and is driven by `rnn_args.yaml` and `train_model.py`.
- Evaluation uses `model_training/evaluate_model.py` and requires a Redis‑connected LM (`language_model/language-model-standalone.py`).
- Data path expected: `data/hdf5_data_final/<session>/data_{train|val|test}.hdf5`.

Deliverables
- Validation WER (for your report) and a test CSV for Kaggle submission.

Sequence (high‑level)
1) Colab setup and dependency install
2) Download data to Drive and verify layout
3) Train baseline GRU in Colab
4) Start Redis + 1‑gram LM; evaluate on val (compute WER) and test (create CSV)
5) If Colab limits block training, train on Kaggle and decode in Colab

---

## 1) Colab setup (copy/paste cells)

1A. Mount Drive and set working directory
```bash
# Verify GPU
!nvidia-smi

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Use a stable workspace folder in Drive
%cd /content/drive/MyDrive
!mkdir -p b2txt25
%cd b2txt25
```

1B. Place/point the repo
- If you already have the repo in Drive, set its path and skip cloning.
```bash
# Option A: Reuse existing repo (recommended if already in Drive)
%cd /content/drive/MyDrive/nejm-brain-to-text || echo "Repo not at this path, see Option B"

Option B: Keep workspace clean, symlink or copy your repo under b2txt25
%cd /content/drive/MyDrive/b2txt25
!ln -s "/content/drive/MyDrive/nejm-brain-to-text" ./nejm-brain-to-text
%cd nejm-brain-to-text
```

1C. Install system + Python deps (conflict‑free set for Colab)
```bash
# System deps
!sudo apt-get update -y
!sudo apt-get install -y redis-server cmake build-essential

# Core GPU stack (CUDA 12.1 wheels on Colab)
!pip -q install --upgrade --no-cache-dir \
  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Remove conflicting HuggingFace libs to avoid resolver issues
!pip -q uninstall -y transformers tokenizers huggingface-hub || true

# Align with Colab constraints
!pip -q install --upgrade --no-cache-dir \
  pandas==2.2.2 \
  numpy==2.0.2

# Known-compatible HF trio
!pip -q install --no-cache-dir \
  huggingface-hub==0.34.1 \
  transformers==4.53.0 \
  tokenizers==0.21.2

# Remaining deps (satisfy umap-learn/tsfresh requirements too)
!pip -q install --upgrade --no-cache-dir \
  redis==5.2.1 \
  matplotlib==3.10.1 \
  scipy==1.14.1 \
  scikit-learn==1.6.1 \
  tqdm==4.67.1 \
  g2p_en==2.1.0 \
  h5py==3.11.0 \
  omegaconf==2.3.0 \
  editdistance==0.8.1 \
  accelerate==1.0.1 \
  bitsandbytes==0.43.1

# Install local utils package from the repo root
%cd /content/drive/MyDrive/nejm-brain-to-text
!pip -q install -e .
```

Troubleshooting (installs)
- If you see: “ERROR: file:///content does not appear to be a Python project”, you ran `pip install -e .` outside the repo. `cd` into `…/nejm-brain-to-text` and rerun.
- If HF trio conflicts, fallback: `transformers==4.51.0`, `tokenizers==0.20.1`, `huggingface-hub==0.34.1`.

---

## 2) Data download and verification
```bash
%cd /content/drive/MyDrive/nejm-brain-to-text
!python download_data.py

# Quick check (first 80 lines)
!ls -R data | head -n 80
```

Expected structure
```
data/
├── t15_copyTask.pkl
├── t15_personalUse.pkl
├── hdf5_data_final/
│   ├── t15.2023.08.11/
│   │   └── data_train.hdf5
│   ├── t15.2023.08.13/
│   │   ├── data_train.hdf5
│   │   ├── data_val.hdf5
│   │   └── data_test.hdf5
│   └── ... (many sessions)
└── t15_pretrained_rnn_baseline/
    └── checkpoint/best_checkpoint, args.yaml
```

---

## 3) Train the baseline GRU (Colab)
```bash
%cd /content/drive/MyDrive/nejm-brain-to-text/model_training

# Optional: quick pipeline check (reduce batches), then restore to 120000 for full run
from omegaconf import OmegaConf
args = OmegaConf.load('rnn_args.yaml')
args.num_training_batches = 120000   # e.g., set 10000 to sanity‑check end‑to‑end
args.gpu_number = '0'
args.output_dir = 'trained_models/baseline_rnn'
args.checkpoint_dir = 'trained_models/baseline_rnn/checkpoint'
OmegaConf.save(config=args, f='rnn_args.yaml')

# Start training
!python train_model.py
```

Outputs (watch for)
- `trained_models/baseline_rnn/training_log`
- `trained_models/baseline_rnn/checkpoint/best_checkpoint`
- `trained_models/baseline_rnn/checkpoint/val_metrics.pkl`

Time
- Full 120k batches: fast GPUs ~3.5h; T4 may be longer. Use smaller `num_training_batches` if session time is tight; resume later by pointing to your last checkpoint (set `init_from_checkpoint: true` and `init_checkpoint_path`).

---

## 4) Language model + evaluation (Colab)

4A. Start Redis
```bash
!redis-server --daemonize yes
!redis-cli ping  # expect PONG
```

4B. Start the 1‑gram LM (keep this cell running)
```bash
%cd /content/drive/MyDrive/nejm-brain-to-text
!python language_model/language-model-standalone.py \
  --lm_path language_model/pretrained_language_models/openwebtext_1gram_lm_sil \
  --do_opt --nbest 100 --acoustic_scale 0.325 --blank_penalty 90 --alpha 0.55 \
  --redis_ip localhost --gpu_number 0
```

Notes
- First run downloads OPT‑6.7b (~13GB). If VRAM is tight on T4, you can remove `--do_opt` (disables OPT rescoring; accuracy drops but RAM usage improves).

4C. Evaluate on validation (compute WER)
```bash
%cd /content/drive/MyDrive/nejm-brain-to-text/model_training
!python evaluate_model.py \
  --model_path trained_models/baseline_rnn \
  --data_dir ../data/hdf5_data_final \
  --eval_type val \
  --gpu_number 0
```

4D. Evaluate on test (produce submission CSV)
```bash
!python evaluate_model.py \
  --model_path trained_models/baseline_rnn \
  --data_dir ../data/hdf5_data_final \
  --eval_type test \
  --gpu_number 0

# Output file lives under the model_path directory:
# trained_models/baseline_rnn/baseline_rnn_test_predicted_sentences_YYYYMMDD_HHMMSS.csv
```

4E. Shutdown Redis (after you finish)
```bash
!redis-cli shutdown
```

---

## 5) Kaggle fallback (if Colab training is too slow/unstable)

Strategy
- Train RNN on Kaggle (GPU notebook). Kaggle may not allow Redis; skip LM there.
- Download the trained checkpoint to Drive.
- Run LM + `evaluate_model.py` in Colab to create submission CSV.

Kaggle steps (high‑level)
1) Create a GPU notebook; attach the repo as a Kaggle Dataset (or upload zip).
2) `pip install` the same Python deps (skip Redis/LM).
3) Run `model_training/train_model.py`; save `trained_models/baseline_rnn/checkpoint/best_checkpoint` to the notebook output.
4) Download the model directory; put it in Drive under `…/model_training/trained_models/baseline_rnn`.
5) Back in Colab, run Section 4 (LM + evaluation) to generate CSV.

---

## Troubleshooting quick reference

- Dependency conflicts (transformers/tokenizers/hf‑hub)
  - Uninstall first, then install compatible trio: `transformers==4.53.0`, `tokenizers==0.21.2`, `huggingface-hub==0.34.1`.
- Colab package constraints
  - Use `pandas==2.2.2`, `numpy==2.0.2`, `scipy==1.14.1`, `scikit-learn==1.6.1`.
- `pip install -e .` error
  - Ensure you `cd` to `/content/drive/MyDrive/nejm-brain-to-text` before installing.
- LM OOM on T4
  - Remove `--do_opt` to skip OPT; or use Colab Pro (A100) for more VRAM.
- Session limits
  - Lower `num_training_batches` to checkpoint quickly; resume later.

---

## Timeline (1 month)

- Week 1: Environment + data + short training (10k batches) to validate end‑to‑end; produce a test CSV.
- Week 2: Full training (120k) on Colab or Kaggle; checkpoint and verify val WER.
- Week 3: Iterate hyperparameters if time allows; re‑evaluate; generate improved CSV.
- Week 4: Final run, prepare report (include val WER, method, settings) and submit to Kaggle.

---

Owner checklist
- Data present under `data/hdf5_data_final/*`
- `trained_models/baseline_rnn/checkpoint/best_checkpoint` exists after training
- LM process connected to Redis (shows “Successfully connected…”)
- Validation WER printed
- Submission CSV generated under model directory

Document version: 2.0

### Option 1: Local Machine (macOS - Current Setup)

**Pros:**
- ✅ No setup required, already have the code
- ✅ Full control over environment
- ✅ No internet dependency during training
- ✅ Easy to iterate and debug

**Cons:**
- ❌ **No GPU support on macOS** (Apple Silicon uses Metal, not CUDA)
- ❌ Training will be extremely slow on CPU (days/weeks)
- ❌ Language model inference requires GPU with 12.4GB+ VRAM
- ❌ Large language models (3gram/5gram) require massive RAM

**Verdict:** ❌ **NOT RECOMMENDED** for training. Only use for code development and testing.

**Recommendation:** Use this setup for code development, then train on a GPU-enabled platform.

---

### Option 2: WSL2 (Windows Subsystem for Linux) with NVIDIA GPU

**Pros:**
- ✅ Native Linux environment (Ubuntu 22.04 recommended)
- ✅ Direct GPU access if NVIDIA GPU is available
- ✅ Can run on Windows machine
- ✅ Full control over environment
- ✅ No cloud costs

**Cons:**
- ❌ Requires Windows 11 with WSL2
- ❌ Requires NVIDIA GPU with CUDA support
- ❌ Requires NVIDIA drivers for WSL2
- ❌ Setup complexity (GPU passthrough)
- ❌ Limited by local hardware resources

**Setup Requirements:**
1. Windows 11 with WSL2 installed
2. NVIDIA GPU with CUDA support (RTX series recommended)
3. NVIDIA drivers for WSL2
4. Ubuntu 22.04 distribution in WSL2

**Estimated Cost:** Free (uses existing hardware)

**Best For:** Users with Windows machine + NVIDIA GPU

**Setup Steps:**
```bash
# Install WSL2 with Ubuntu 22.04
wsl --install -d Ubuntu-22.04

# Install NVIDIA drivers for WSL2
# Download from: https://www.nvidia.com/Download/index.aspx

# Inside WSL2, install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda-repo-wsl-ubuntu-12-6-local_12.6.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-6-local_12.6.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Verify CUDA
nvidia-smi
nvcc --version

# Clone repository and setup
git clone <repo-url>
cd nejm-brain-to-text
./setup.sh
./setup_lm.sh
```

---

### Option 3: Google Colab (Free Tier)

**Pros:**
- ✅ Free GPU access (T4, 16GB VRAM)
- ✅ Pre-configured environment
- ✅ No local setup required
- ✅ Easy to share and collaborate
- ✅ Jupyter notebook interface

**Cons:**
- ❌ **Limited runtime** (12 hours max, then disconnects)
- ❌ **Training may not complete** in one session (3.5+ hours needed)
- ❌ Unstable connection (can disconnect)
- ❌ Limited storage (need to upload/download data)
- ❌ Can't run Redis server easily
- ❌ Difficult to run language model pipeline
- ❌ No guarantee of GPU availability

**Estimated Cost:** Free (with limitations)

**Best For:** Quick experiments, testing code, prototyping

**Setup Steps:**
1. Upload project to Google Drive or GitHub
2. Mount Google Drive in Colab
3. Install dependencies in Colab notebook
4. Run training (may need multiple sessions)

**Limitations:**
- Training time: ~3.5 hours (but session limits: 12 hours max)
- May need to save checkpoints and resume
- Language model evaluation is complex in Colab

**Verdict:** ⚠️ **POSSIBLE BUT CHALLENGING** - Good for prototyping, not ideal for full pipeline

---

### Option 4: Google Colab Pro ($10/month)

**Pros:**
- ✅ More reliable GPU access (T4, A100 options)
- ✅ Longer runtime sessions
- ✅ Better performance
- ✅ Priority access to GPUs

**Cons:**
- ❌ Still has session limits
- ❌ Monthly subscription cost
- ❌ Storage limitations
- ❌ Complex setup for full pipeline

**Estimated Cost:** $10/month

**Best For:** Users who want better Colab experience

**Verdict:** ⚠️ **BETTER THAN FREE TIER** but still has limitations

---

### Option 5: Kaggle Notebooks (Free)

**Pros:**
- ✅ Free GPU access (P100, 16GB VRAM)
- ✅ 30 hours/week GPU time limit
- ✅ Pre-configured environment
- ✅ Competition-specific platform
- ✅ Easy data access
- ✅ Can run for ~9 hours per session

**Cons:**
- ❌ Limited to 30 hours/week GPU time
- ❌ Session limits (~9 hours max)
- ❌ Storage limitations
- ❌ Complex setup for Redis + language model
- ❌ Internet access restrictions

**Estimated Cost:** Free

**Best For:** Competition participants, quick experiments

**Setup Approach:**
1. Upload project as Kaggle dataset
2. Create new notebook with GPU enabled
3. Install dependencies
4. Run training (may need to save/resume)

**Verdict:** ⚠️ **GOOD FOR COMPETITION** but may need multiple sessions for full training

---

### Option 6: AWS EC2 (Recommended)

**Pros:**
- ✅ Full control over environment
- ✅ Choose GPU instance (g4dn, p3, p4d)
- ✅ Can run complete pipeline
- ✅ Persistent storage (EBS)
- ✅ Can run 24/7 if needed
- ✅ Professional setup

**Cons:**
- ❌ Costs money ($0.50-$10+/hour depending on instance)
- ❌ Requires AWS account setup
- ❌ Need to manage instance lifecycle
- ❌ More complex initial setup

**Estimated Cost:** 
- **g4dn.xlarge** (T4, 16GB): ~$0.50/hour = **~$1.75 per training run**
- **p3.2xlarge** (V100, 16GB): ~$3.06/hour = **~$10.71 per training run**
- **p4d.24xlarge** (A100, 40GB): ~$32.77/hour = **~$114.70 per training run**

**Best For:** Serious training, production workloads

**Recommended Instance:** `g4dn.xlarge` (T4 GPU, 16GB VRAM, sufficient for training)

**Setup Steps:**
```bash
# 1. Launch EC2 instance
# - AMI: Ubuntu 22.04 LTS
# - Instance: g4dn.xlarge (or larger)
# - Storage: 100GB+ (for data and models)
# - Security Group: Allow SSH (port 22)

# 2. Connect via SSH
ssh -i your-key.pem ubuntu@your-instance-ip

# 3. Install NVIDIA drivers and CUDA
sudo apt-get update
sudo apt-get install -y build-essential
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda-repo-ubuntu2204-12-6-local_12.6.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-6-local_12.6.0-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
sudo apt-get -y install nvidia-driver-550

# 4. Install Redis, CMake, GCC
sudo apt-get install -y redis-server build-essential cmake
sudo systemctl disable redis-server

# 5. Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
source ~/miniconda3/bin/activate

# 6. Clone repository and setup
git clone <repo-url>  # or upload via SCP
cd nejm-brain-to-text
./setup.sh
./setup_lm.sh

# 7. Download data
conda activate b2txt25
python download_data.py

# 8. Train model (use screen or tmux for long-running jobs)
screen -S training
conda activate b2txt25
cd model_training
python train_model.py
# Press Ctrl+A then D to detach

# 9. Monitor training
screen -r training

# 10. Download results when done
# Use SCP to download trained models
scp -i your-key.pem -r ubuntu@your-instance-ip:~/nejm-brain-to-text/trained_models ./
```

**Cost Optimization Tips:**
- Use Spot Instances for 70% cost savings (but can be interrupted)
- Stop instance when not training
- Use smaller instance for evaluation only
- Consider Reserved Instances for long-term use

**Verdict:** ✅ **HIGHLY RECOMMENDED** for serious training

---

### Option 7: Google Cloud Platform (GCP)

**Pros:**
- ✅ Similar to AWS, full control
- ✅ Good GPU options
- ✅ $300 free credits for new users
- ✅ Persistent storage

**Cons:**
- ❌ Costs money after free credits
- ❌ More complex setup
- ❌ Need GCP account

**Estimated Cost:**
- **n1-standard-4 + T4 GPU**: ~$0.35/hour = **~$1.23 per training run**
- **n1-standard-8 + V100**: ~$2.50/hour = **~$8.75 per training run**

**Best For:** Users with GCP credits or preference

**Setup:** Similar to AWS, but using GCP Compute Engine

**Verdict:** ✅ **GOOD ALTERNATIVE TO AWS**

---

### Option 8: Azure ML / Azure Compute

**Pros:**
- ✅ Managed ML platform
- ✅ Good GPU options
- ✅ $200 free credits for new users
- ✅ Integration with ML tools

**Cons:**
- ❌ Costs money
- ❌ More complex setup
- ❌ Need Azure account

**Estimated Cost:** Similar to AWS/GCP

**Verdict:** ✅ **GOOD OPTION** if you prefer Azure ecosystem

---

### Option 9: Lambda Labs / Vast.ai / RunPod (GPU Rental)

**Pros:**
- ✅ Cheaper than AWS/GCP (often 50-70% less)
- ✅ Pay per hour
- ✅ Good GPU selection
- ✅ Simple setup

**Cons:**
- ❌ Less established providers
- ❌ May have less reliability
- ❌ Need to trust third-party

**Estimated Cost:**
- **RTX 3090 (24GB)**: ~$0.35/hour = **~$1.23 per training run**
- **A100 (40GB)**: ~$1.10/hour = **~$3.85 per training run**

**Best For:** Cost-conscious users

**Verdict:** ✅ **COST-EFFECTIVE OPTION**

---

### Option 10: University/Research Compute Cluster

**Pros:**
- ✅ Often free for students/researchers
- ✅ High-performance GPUs
- ✅ Professional infrastructure
- ✅ Support available

**Cons:**
- ❌ May require approval/access
- ❌ May have usage limits
- ❌ Less control
- ❌ May have queue waiting times

**Best For:** Students with access to university resources

**Verdict:** ✅ **BEST IF AVAILABLE**

---

## 🎯 Platform Recommendation Summary

### For Training (Ranked by Preference):

1. **🥇 University Compute Cluster** (if available)
   - Free, high-performance, professional setup

2. **🥈 AWS EC2 (g4dn.xlarge)** 
   - ~$1.75 per training run
   - Full control, reliable, professional

3. **🥉 Lambda Labs / Vast.ai**
   - ~$1.23 per training run (RTX 3090)
   - Cost-effective, good performance

4. **Kaggle Notebooks**
   - Free, but limited to 30 hours/week
   - Good for competition, may need multiple sessions

5. **Google Colab Pro**
   - $10/month, but still has limitations
   - Good for prototyping

6. **WSL2 (if you have NVIDIA GPU)**
   - Free, but requires Windows + NVIDIA GPU
   - Good if hardware is available

7. **Local macOS**
   - ❌ Not recommended (no CUDA support)

---

## 📊 Resource Requirements Summary

### Minimum Requirements:
- **GPU**: NVIDIA GPU with CUDA support (8GB+ VRAM minimum, 16GB+ recommended)
- **RAM**: 16GB minimum (60GB+ for 3gram LM, 300GB+ for 5gram LM)
- **Storage**: 50GB+ for data and models
- **OS**: Ubuntu 22.04 (recommended) or Linux equivalent
- **Training Time**: ~3.5 hours on RTX 4090, longer on slower GPUs

### Recommended Setup:
- **GPU**: RTX 3090/4090, V100, or A100 (16GB+ VRAM)
- **RAM**: 32GB+ (64GB+ for 3gram LM)
- **Storage**: 100GB+ SSD
- **OS**: Ubuntu 22.04 LTS
- **Network**: Stable connection for data download

---

## ⚠️ Important Considerations

1. **Training Interruptions**: 
   - Save checkpoints regularly (configured in `rnn_args.yaml`)
   - Use `screen` or `tmux` for long-running jobs
   - Consider resumable training if interrupted

2. **Data Storage**:
   - Data is ~10GB+ compressed
   - Unzipped data is larger
   - Trained models are several GB
   - Plan for sufficient storage

3. **Language Model Requirements**:
   - OPT 6.7b requires 12.4GB+ VRAM
   - 3gram LM requires ~60GB RAM
   - 5gram LM requires ~300GB RAM
   - May need to use smaller models or upgrade hardware

4. **Redis Server**:
   - Required for language model inference
   - Must run during evaluation
   - Can run on same machine or separate instance

5. **Multiple Environments**:
   - Two conda environments needed (`b2txt25` and `b2txt25_lm`)
   - Different PyTorch versions (training vs. LM)
   - Cannot mix environments

---

## 🚀 Quick Start Recommendation

**For Quick Testing:**
1. Use **Kaggle Notebooks** (free, 30 hours/week)
2. Upload project as dataset
3. Run training in notebook
4. Save checkpoints and download results

**For Serious Training:**
1. Use **AWS EC2 g4dn.xlarge** (~$1.75 per run)
2. Follow AWS setup steps above
3. Use `screen` for long-running training
4. Download results when complete

**For Cost-Conscious:**
1. Use **Lambda Labs** or **Vast.ai** (~$1.23 per run)
2. Similar setup to AWS
3. Monitor usage carefully

---

## ❓ Questions to Clarify

Before proceeding, please confirm:

1. **What is your primary goal?**
   - [ ] Just get baseline model running
   - [ ] Compete in Kaggle competition
   - [ ] Experiment with improvements
   - [ ] Reproduce paper results

2. **What resources do you have access to?**
   - [ ] University compute cluster
   - [ ] Local machine with NVIDIA GPU
   - [ ] AWS/GCP/Azure account
   - [ ] Budget for cloud computing ($1-5 per training run)
   - [ ] Only free options

3. **What is your timeline?**
   - [ ] Need results ASAP
   - [ ] Can wait for free resources
   - [ ] Have weeks/months

4. **What is your experience level?**
   - [ ] Comfortable with Linux/cloud setup
   - [ ] Prefer managed platforms (Colab/Kaggle)
   - [ ] Need step-by-step guidance

5. **Do you need the full pipeline?**
   - [ ] Just training the RNN model
   - [ ] Need language model evaluation too
   - [ ] Need to generate submission file

---

## 📝 Next Steps

Once you've chosen a platform:

1. **Confirm platform choice**
2. **Set up environment** (follow platform-specific steps)
3. **Download and verify data**
4. **Run training** (monitor closely first time)
5. **Evaluate model** (validation set first)
6. **Generate submission** (test set)
7. **Submit to Kaggle**

---

**Document Version:** 1.0  
**Last Updated:** 2025-01-XX  
**Author:** Training Plan Generator

