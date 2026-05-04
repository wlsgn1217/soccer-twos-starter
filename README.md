# Soccer-Twos Training Pipeline

Three-stage pipeline for training a privileged teacher policy and distilling it into a student policy via DAgger.

```
Stage 1: Curriculum → Privileged Teacher (PPO + curriculum)
Stage 2: Baseline   → Fine-tune Teacher vs CEIA opponent
Stage 3: DAgger     → Distill Teacher into Student (no privileged obs at inference)
```

---

## Setup

### Requirements

- Python 3.8
- See [requirements.txt](requirements.txt)

### Installation

```bash
# 1. Fork and clone
git clone https://github.com/your-github-user/soccer-twos-starter.git
cd soccer-twos-starter/

# 2. Create conda environment
conda create --name soccertwos python=3.8 -y
conda activate soccertwos

# 3. Downgrade build tools for compatibility
pip install pip==23.3.2 setuptools==65.5.0 wheel==0.38.4
pip cache purge

# 4. Install requirements
pip install -r requirements.txt

# 5. Fix protobuf and pydantic compatibility
pip install protobuf==3.20.3
pip install pydantic==1.10.13
```

### Baseline Agent

Download the CEIA baseline agent and extract it to the project root:

[Download ceia_baseline_agent](https://drive.google.com/file/d/1WEjr48D7QG9uVy1tf4GJAZTpimHtINzE/view?usp=sharing)

```bash
# After extracting, verify:
ls ceia_baseline_agent/
```

---

## Stage 1 — Curriculum Training (Privileged Teacher)

Train the teacher policy with a privileged observation vector (full ground-truth state) using progressive curriculum learning.

**Curriculum config:** `curriculum_for_privileged.yaml` — 5 stages from easy goal-scoring scenarios to random full-field play.

```bash
python train_ray_curriculum_privileged.py
```

The script auto-advances curriculum stages when `episode_reward_mean > 1.5` and stops at `1.9` or after 150M timesteps / 20 hours.

**Output:** `ray_results/PPO_curriculum/<run>/checkpoint_<N>/`

At the end, split weights are automatically exported to `split_stage1/` inside the best checkpoint directory:
- `split_stage1/actor_weights.pt`
- `split_stage1/privileged_encoder_weights.pt`

To export manually from an existing checkpoint:
```bash
python tools/export_stage1_privileged_weights.py \
    --checkpoint ./ray_results/PPO_curriculum/<run>/checkpoint_<N>/checkpoint-<N> \
    --out-dir ./ray_results/PPO_curriculum/<run>/checkpoint_<N>/split_stage1
```

---

## Stage 2 — Fine-tune Teacher Against Baseline

Fine-tune the privileged teacher policy by playing against the fixed CEIA baseline agent.

### Option A: Direct fine-tuning (no curriculum)

```bash
python train_against_baseline.py \
    --init-policy-checkpoint ./ray_results/PPO_curriculum/<run>/checkpoint_<N>/checkpoint-<N>
```

Key options:
| Flag | Default | Description |
|------|---------|-------------|
| `--init-policy-checkpoint` | AUTO (picks latest curriculum checkpoint) | Warm-start checkpoint from Stage 1 |
| `--restore-checkpoint` | `` | Resume an interrupted baseline run |
| `--prox-weight` | `0.3` | Reward shaping: agent-to-ball proximity |
| `--goal-weight` | `0.5` | Reward shaping: ball progress toward goal |
| `--obs-mode` | `privileged_dict` | Use `privileged_dict` for teacher; `raw` for FC net |
| `--run-tag` | `` | Label added to output directory name |

### Option B: Curriculum fine-tuning against baseline

Combines structured scenario sampling with baseline play (12 progressive stages).

**Curriculum config:** `curriculum_for_vs_baseline.yaml`

```bash
python train_curriculum_against_baseline.py \
    --init-policy-checkpoint ./ray_results/PPO_curriculum/<run>/checkpoint_<N>/checkpoint-<N> \
    --curriculum curriculum_for_vs_baseline.yaml
```

Key options (same as Option A, plus):
| Flag | Default | Description |
|------|---------|-------------|
| `--curriculum` | `curriculum_for_vs_baseline.yaml` | Curriculum YAML defining scenario stages |
| `--goal-weight` | `0.7` | Reward shaping for goal progress |

**Output:** `ray_results/<run_label>_<timestamp>/` with RLlib checkpoints.

Export split weights from the best checkpoint after training:
```bash
python tools/export_stage1_privileged_weights.py \
    --checkpoint ./ray_results/<run>/checkpoint_<N>/checkpoint-<N> \
    --out-dir ./split_stage1
```

---

## Stage 3 — DAgger Student Distillation

Distill the teacher into a student that does **not** require privileged observations at inference time. Two sub-stages: first clone the privileged encoder, then replace it with a GRU history encoder.

### Stage 3a — Privileged Student (DAgger with privileged obs)

The student receives `(obs, privileged_vec)` — same inputs as teacher — and is trained via behavior cloning with DAgger beta-mixing.

```bash
python train_dagger_privileged.py \
    --expert-module ceia_baseline_agent \
    --out-dir ./dagger_artifacts \
    --dagger-iters 20 \
    --episodes-per-iter 20
```

Key options:
| Flag | Default | Description |
|------|---------|-------------|
| `--expert-module` | `ceia_baseline_agent` | Teacher agent module |
| `--out-dir` | `./dagger_artifacts` | Output directory for dataset + weights |
| `--dagger-iters` | `10` | Number of DAgger iterations |
| `--episodes-per-iter` | `20` | Episodes collected per iteration |
| `--beta-start` / `--beta-end` | `1.0` / `0.05` | Teacher mixing schedule (1.0 = always teacher) |
| `--init-split-dir` | — | Warm-start from prior `split_stage1/` directory |
| `--pack-rllib-checkpoint` | off | Pack output weights into an RLlib checkpoint shell |

**Output:**
```
dagger_artifacts/
  dataset_iter_000.npz ... dataset_iter_N.npz
  split_stage1/
    actor_weights.pt
    privileged_encoder_weights.pt
```

### Stage 3b — History Encoder Student (GRU, no privileged obs at inference)

Replaces the privileged encoder with a GRU that reads a rolling window of `(obs_t, prev_action_t)` pairs. The actor backbone is frozen; only the GRU is trained.

Loss: `alpha * MSE(gru_latent, privileged_latent) + (1-alpha) * KL(student_logits, teacher_logits)`

```bash
python train_dagger_student.py \
    --stage1-split-dir ./dagger_artifacts/split_stage1 \
    --out-dir ./dagger_student_run1
```

Key options:
| Flag | Default | Description |
|------|---------|-------------|
| `--stage1-split-dir` | *required* | Directory with Stage 3a `split_stage1/` weights |
| `--out-dir` | `./dagger_student_artifacts` | Output directory |
| `--baseline-module` | `ceia_baseline_agent` | Opponent for trajectory collection |
| `--window-len` | `20` | GRU BPTT window length |
| `--dagger-iters` | `20` | Number of DAgger iterations |
| `--alpha` | `0.5` | Weight on latent MSE vs action KL loss |
| `--gru-hidden-size` | `128` | GRU hidden state dimension |
| `--worker-id` | `0` | ML-Agents worker ID (use different values per parallel run) |
| `--base-port` | `5005` | ML-Agents base port |

**Output:**
```
dagger_student_run1/
  history_encoder_iter_000.pt ... history_encoder_iter_N.pt
  history_encoder_final.pt
  history_encoder_meta.pt
```

---

## Full Pipeline

```bash
# Stage 1: curriculum teacher
python train_ray_curriculum_privileged.py
# → ray_results/PPO_curriculum/<run>/checkpoint_<N>/split_stage1/

# Stage 2: fine-tune against baseline (curriculum variant)
python train_curriculum_against_baseline.py \
    --init-policy-checkpoint ./ray_results/PPO_curriculum/<run>/checkpoint_<N>/checkpoint-<N>
# Then export split weights:
python tools/export_stage1_privileged_weights.py \
    --checkpoint ./ray_results/<run>/checkpoint_<N>/checkpoint-<N> \
    --out-dir ./split_stage1

# Stage 3a: privileged student via DAgger
python train_dagger_privileged.py \
    --out-dir ./dagger_artifacts
# → dagger_artifacts/split_stage1/

# Stage 3b: GRU history-encoder student
python train_dagger_student.py \
    --stage1-split-dir ./dagger_artifacts/split_stage1 \
    --out-dir ./dagger_student_run1
# → dagger_student_run1/history_encoder_final.pt
```

---

## File Structure

```
soccer-twos-starter/
├── train_ray_curriculum_privileged.py   # Stage 1: curriculum PPO for privileged teacher
├── train_against_baseline.py            # Stage 2a: teacher vs CEIA baseline (direct)
├── train_curriculum_against_baseline.py # Stage 2b: teacher vs CEIA baseline (with curriculum)
├── train_dagger_privileged.py           # Stage 3a: DAgger privileged student
├── train_dagger_student.py              # Stage 3b: DAgger history-encoder student (GRU)
│
├── custom_env_wrapper.py                # Env wrapper (privileged obs, reward shaping)
├── privileged_features.py              # Privileged observation builder (25-d vector)
├── curriculum_sampling.py              # YAML curriculum state sampling
├── privileged_curriculum_sampling.py   # Stage 1 curriculum config + thresholds
├── dagger_utils.py                     # DAgger dataset, beta schedule, expert loader
├── utils.py                            # RLlib env registration helpers
│
├── curriculum_for_privileged.yaml       # Stage 1 curriculum definition (5 stages)
├── curriculum_for_vs_baseline.yaml      # Stage 2b curriculum definition (12 stages)
│
├── models/
│   ├── privileged_actor_model.py        # RLlib teacher model (privileged encoder + actor)
│   ├── history_encoder.py              # GRU history encoder (Stage 3b)
│   ├── gru_student_model.py            # Full GRU student model wrapper
│   ├── _mlp.py                         # MLP builder helper
│   └── rllib_registration.py           # Register custom models with RLlib
│
├── tools/
│   ├── export_stage1_privileged_weights.py  # Export actor+encoder weights from checkpoint
│   └── pack_split_to_rllib_checkpoint.py    # Repack split weights into RLlib checkpoint
│
└── ceia_baseline_agent/                # Fixed CEIA opponent (required for Stages 2-3)
```

---

## Agent Submission

To submit an agent, implement a class inheriting from `soccer_twos.AgentInterface` with an `act` method. Examples are in `example_player_agent/` or `example_team_agent/`. Compress the agent module folder as `.zip` for submission.

Test your agent:
```bash
python -m soccer_twos.watch -m your_agent_folder
python -m soccer_twos.watch -m1 your_agent_folder -m2 ceia_baseline_agent
```
