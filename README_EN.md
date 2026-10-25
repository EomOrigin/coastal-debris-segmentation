[한국어](README.md) | **English**

# Coastal Debris Segmentation from Satellite Imagery — Trained Only on Synthetic Data

Code for my final submission to AIFactory's "2026 National Park Satellite Monitoring AI Challenge", Topic 4 (coastal debris detection and extent estimation). I entered solo.
The final result was 6th of 138 teams on the private leaderboard (0.8586).

## Overview

| Item | Details |
|---|---|
| Competition | AIFactory #9307, 2026 National Park Satellite Monitoring AI Challenge, Topic 4: Coastal debris detection and extent estimation (2026-09-14 to 10-06) |
| Task | Single-class segmentation of 0.3 m-class 256×256 RGB satellite patches; submit a mask of debris-zone pixels as RLE |
| Metric | 0.5 × presence macro-F1 (a mask with ≥ 50 pixels counts as positive) + 0.5 × shape score (mean 1-pixel-tolerant F over ground-truth positive patches; a missed positive scores 0) |
| Team | 다닐로바 (solo) |
| Result | **Private 0.8586483812, 6th of 138 teams** · Public 0.8320671333 (9th at submission time) |
| Final submission | `newbase-v22ens3` (`submission/` in this repository) |

## The problem

- The organizer provided no training data at all. The evaluation images are only available on the scoring server, where the submitted notebook runs inference automatically. I had to build all training data myself.
- The sensor and processing behind the evaluation images were unknown. My first model (v1) scored 0.902 on my own synthetic test set but 0.651 on the public leaderboard, a gap of 0.25.
- Diagnosis showed v1 was a "sharp colored dot detector". On synthetic positives with only white and gray pieces it dropped to 0.263. In my synthesizer, positives always contained colored pieces while fake negatives (gravel, foam) were achromatic, so I concluded it had learned "colored means debris". Breaking that shortcut in v4 raised the public score by +0.122.
- Only 3 scored submissions per day, and each returns a single number. The public set has 194 patches (124 positive, 70 negative), which I recovered by inverting public leaderboard scores through the metric formula. With so few patches, the standard deviation of a score difference between two submissions was estimated at 0.015–0.02, so small improvements disappear in noise.
- The metric punishes false negatives (FN) more than false positives (FP). A missed positive loses both F1 and that patch's shape score; a false positive only loses F1. In the final state, losing one true positive cost −0.0058 and removing one false positive gained +0.0033. A rule that filters false positives only pays off if it removes more than 1.8 false positives per true positive it loses.

## Method

### 1. Backgrounds and synthetic data (`src/synth.py`, `data/scripts/`)

- **Backgrounds**: 3,950 scenes of 1024×1024 at 0.3 m, centered on OSM coastline points, from five open orthoimagery sources (USGS NAIP and NOAA ERI in the US, LINZ in New Zealand, GSI in Japan, IGN in France). 8% per source and region (317 scenes) is held out as a test split that no model trains on. The training list (3,679 scenes, including 46 collected earlier) is split by scene into 2,944 train / 735 validation.
- **Debris synthesis**: on a background crop I build a density field shaped as bands that follow the shoreline (length 30–260 px, width 3–18 px) or clumps (radius 5–26 px), render pieces on a 4× supersampled canvas, and downsample. The ground truth is every pixel with piece coverage ≥ 0.5. The positive rate is 0.64.
- **Breaking the color shortcut**: 25% of debris zones use achromatic pieces only, 50% use a colored fraction of 5–45%. After synthesis, only the chroma (Cb/Cr) is blurred with σ 0.3–2.5 (probability 0.7), and JPEG (q 70–95, probability 0.3) is mixed in.
- **Satellite look**: 60% of samples get darker, bluer water and a scene-level 2–98% stretch. The target was the mean color of the evaluation images (RGB 48/59/64), estimated from the BN statistics of the organizer's baseline weights.
- **Negatives**: foam, gravel and shells, wrack and driftwood, puddles, and aquaculture buoy lines are synthesized with no ground truth. 4,667 real-background crops that an early model (v2) wrongly flagged as debris make up 20% of negatives.
- **Training augmentation**: grayscale (probability 0.2), hue (±0.05) and saturation (×0.5–1.3) jitter.

### 2. Separate models for "is there debris" and "where is it"

The final model splits the presence decision and the mask between different models.
Keeping the presence decision fixed and swapping only the mask for larger models raised the public score (v13a, +0.013), while letting large models also make the presence decision lowered it.

| Stage | Model | Rule |
|---|---|---|
| Presence (stage 1) | U-Net (ResNet-34) with a presence head, `v4_r34`, input upscaled 2× | Positive if the mean presence probability over 8 rotations/flips ≥ 0.9 |
| Rescue | Frozen DINOv3 segmentation model `v19_d3` | Positive if presence is 0.7–0.9 and at least 50 pixels have mask probability ≥ 0.9 |
| Verifier (stage 2) | Frozen DINOv2-B/14 reg4 features (448 input) + MLP | Re-checks stage-1 positives only; negative if p < 6.14e-7 |
| Mask | Three frozen DINOv3 satellite ViT-L/16 segmentation models (`v22p`, `v22p_s2`, `v26mk2`) | 8-view TTA probabilities averaged 1/3 each, threshold 0.35 |
| Post-processing | — | A positive with a mask under 50 pixels is grown to 56 pixels from the highest-probability pixels; a negative gets an empty mask |

- **Mask model** (`src/vfm_net.py`): DINOv3 ViT-L/16 SAT-493M (pretrained on 493M satellite images) stays frozen. Each block gets a Rein-style adapter (100 learnable tokens, gate initialized at 0) and a per-position low-rank term (rank 64). A pyramid decoder uses features from 4 layers, plus a full-resolution CNN branch. A frozen AnyUp upsampler produces full-resolution features that are added at the stride-2 stage. Input is 768 px (token grid 48); 10.5M of 314.4M parameters are trained.
- **Loss** (`src/train.py`): BCE excluding the 1-pixel ring outside the ground truth + a differentiable surrogate of the 1-pixel-tolerant F + 0.5·Dice + 0.5·presence BCE. The three mask models lower the positive pixel weight from 3 to 1 and use F_β (β 0.8) to lean toward precision, because precision measured on the public set (0.71) was below recall (0.82), meaning the masks over-painted.
- **The three mask models**: starting from `v19_d3` (512 px, 6,000 steps), training continued at 768 px for `v22p` (4,000 steps, seed 1) and `v22p_s2` (1,500 steps, seed 2). `v26mk2` fine-tunes `v22p_s2` for 1,500 steps with recipe v26, which replaces half of the hard negatives with scattered pieces that do not form a debris zone and places them, without ground truth, on both negative and positive patches. So `v26mk2` leans toward painting only debris zones.
- **Verifier** (`scripts/verifier/`): trained only on samples that fool the stage-1 model. 21,369 negatives (12,376 mined real-coast crops, 3,141 Dutch tidal flats, 3,332 AI-Hub Taean drone tiles, 2,520 synthetic confusers) and 9,870 positives (6,824 synthetic debris zones, 2,901 zones built from AI-Hub 506 pieces, 145 real AI-Hub Taean tiles). τ keeps 99.5% of positives on a validation split grouped by scene.
- **Inference safeguards** (`submission/predict.ipynb`): if the run is projected to exceed the 5,700 s budget, only the mask TTA is reduced from 8 to 2 to 1 views; the presence path always uses 8 views. If a mask model fails, that batch falls back to the presence model's own mask.

## Validation strategy

1. **Synthetic validation set**: checkpoints and thresholds were chosen with the competition metric on 1,000 synthetic validation patches (valmix: six appearance groups — base, chroma blur σ1.5 and σ2.0, satellite look, achromatic, desaturated).
2. **Limits of the proxy**: at first, the "chroma blur σ1.5" panel predicted the public order three times in a row (v1 < v6 < v4). Then it predicted +0.033 for a large backbone (v10) and the public score moved −0.007.
3. **Checking the validation sets**: I re-scored 7 submissions with known public scores, as packaged, on 10 validation sets. valmix got all 13 pairs with a public difference ≥ 0.015 right, but no synthetic set could separate differences of about ±0.01. Because the validation variations were also training augmentations, the synthetic score measured how well a model fit my synthesizer. The validation-minus-public offset grew from −0.02 (v1) to +0.14 (v11).
4. **Public A/B**: from v13a on, each submission changed either the presence decision or the mask, never both. With the same presence decision, twice the score difference equals the shape score difference.
5. **Diagnostic submissions**: keep the presence decision and replace every positive mask with a 56-pixel dummy that cannot overlap the ground truth. The shape score becomes 0, so score = 0.5 × macro-F1, and the 10-digit score determines a unique (true positive, false positive) pair. The configuration just before the final one (v22pens) had 124 true positives, 22 false positives and 0 misses; its 0.171 gap to a perfect score split into 0.067 from false positives and 0.104 from mask quality.
6. **What it led to**: the remaining 22 false positives were patches that the presence model, DINOv2 and DINOv3 all confidently called debris. Rules that filtered false positives well locally also dropped true positives on the public set and never cleared break-even. The last gain came from the mask side, by mixing in a model with a different bias (`v26mk2`).

## Results

Key public leaderboard milestones (194 public patches; rank at that time in parentheses).

| Date | Version | Change | Public |
|---|---|---|---|
| 9/28 | v1 | First model trained only on synthetic data (U-Net ResNet-34) | 0.6511 (20th/90) |
| 9/29 | v4 | Break the color shortcut + satellite look + mined real-background false positives + metric-aware loss | 0.7732 (16th/92) |
| 9/30 | v13a | Keep v4's presence decision, mask = mean of three large backbones | 0.7861 (12th) |
| 10/2 | v16a | Add the DINOv2 verifier (removed 1 false positive) | 0.7898 |
| 10/3 | v18t90 | Presence threshold 0.4 → 0.9 | 0.8126 (11th) |
| 10/4 | v19t90 | Mask from a frozen DINOv3 satellite model | 0.8161 (10th) |
| 10/4 | v21r70 | Add the rescue rule | 0.8172 (10th) |
| 10/5 | v22pens | 768 px, Rein per-position term, AnyUp, precision-leaning loss; mean of two masks | 0.8287 (9th) |
| 10/6 | **v22ens3 (final)** | Add `v26mk2`, which does not paint scattered pieces; mean of three masks | **0.8321 (9th)** |

On the private leaderboard the final submission scored 0.8586, 6th of 138 teams.

Attempts that did not help:

| Attempt | Public | Reading |
|---|---|---|
| Backgrounds 46 → 3,679 scenes (v2) | 0.6380 (v1 −0.013) | Background diversity was not the bottleneck |
| Presence threshold 0.4 → 0.3 (v4b) | 0.7501 (v4 −0.023) | The 0.3–0.4 band was mostly negatives |
| Add AI-Hub Korean real-debris pieces and backgrounds (v6) | 0.7555 (v4 −0.018) | Moving toward small, achromatic real debris hurt; the evaluation debris looked closer to large multi-colored zones |
| Large backbone EfficientNetV2-XL (v10) / 3-model ensemble (v11) | 0.7659 / 0.7631 | Read as larger models fitting the synthetic distribution more and generalizing worse |
| Shift every mask by 1 pixel (v18s) | shape −0.016 | The misalignment with the ground truth varies in direction per patch |
| Retrain the presence model with scattered-piece negatives (v26b) | 0.8284 (v22pens −0.0003) | About 2 false positives removed per true positive lost: break-even |
| Generalized presence veto rule (v28a) | 0.8223 (v22pens −0.0064) | Local filtering power did not transfer to the public set |

## How to run

Assumes one GPU (an RTX A6000 48 GB during the competition). I ran everything in a conda environment inside a Docker container (Python 3.11, PyTorch 2.5.1 + CUDA 12.1).

```bash
pip install -r requirements.txt          # training and local inference
pip install -r requirements-data.txt     # data collection (can be a separate env)
```

**1) Collect backgrounds** — written to `data/backgrounds/`

```bash
python data/scripts/fetch_bg_v2.py points --shp <lines.shp from coastlines-split-4326>
for s in gsi naip linz ign eri; do python data/scripts/fetch_bg_v2.py fetch --source $s; done
python data/scripts/make_splits_v2.py    # splits/v2_train.txt, v2_test.txt
```

**2) Train the models** — needs `src/pretrained/anyup_paper.pth` (weights from the AnyUp repository). timm and smp download the DINOv3 and ResNet-34 pretrained weights.

```bash
bash scripts/train_final.sh v2        # early model used only for mining
bash scripts/train_final.sh mine_v2   # mine real-background false positives, then check sheet.jpg by eye
bash scripts/train_final.sh v4        # presence model (about 118 min on an A6000, from the logs)
bash scripts/train_final.sh v19_d3    # DINOv3 stage 1, used by the rescue rule (about 123 min)
bash scripts/train_final.sh v22p      # mask 1 (about 143 min)
bash scripts/train_final.sh v22p_s2   # mask 2 (about 56 min)
bash scripts/train_final.sh v26mk2    # mask 3 (about 55 min)
```

**3) Verifier** — download AI-Hub 71363 and 506 yourself and place them as described in `data/README.md`.

```bash
bash scripts/train_final.sh mine_v4
python scripts/verifier/fetch_pdok.py
python data/scripts/build_aihub_debris.py --split train --src <TS_01.Drone.zip>
python data/scripts/extract_kr_bg.py <TS_01.Drone.zip> data/aihub_71363/derived/kr_bg/train
python scripts/verifier/extract_506_sprites.py
python scripts/verifier/export_dinov2.py
python scripts/verifier/build_verifier_data.py          # -> feats.npz
python scripts/verifier/add_506.py                      # -> feats_506.npz
python scripts/verifier/add_506.py merge                # -> feats_all.npz
python scripts/verifier/train_verifier.py --feats feats_all.npz   # -> verifier_head.pt, verifier.json
```

**4) Build the submission package** — copies the weights into `submission/assets/model/` and updates the file names in config.json.

```bash
python scripts/make_package.py \
  --presence src/runs/v4_r34/weights_fp16.pt \
  --mask src/runs/v22p/weights_step4000.pt src/runs/v22p_s2/weights_step1500.pt src/runs/v26mk2/weights_step1500.pt \
  --rescue src/runs/v19_d3/weights_step6000.pt \
  --verifier_backbone scripts/verifier/dinov2_vitb14_reg4_448_fp16.pt \
  --verifier_head scripts/verifier/verifier_head.pt \
  --tau <tau_conservative from verifier.json>
```

**5) Run and score locally** — runs the notebook the way the scoring server does and scores it with the competition metric.

```bash
cd src
python e2e_check.py make e2e/input ../data/backgrounds/splits/v2_test.txt 300   # synthetic test set (ground truth kept separately)
python run_local.py ../submission e2e/input e2e/pred.csv
python e2e_check.py score e2e/input e2e/pred.csv
```

During the competition I submitted from `submission/` with `AIF_API_KEY=<participation key> python -m aifactory submit --notebook predict.ipynb --model-name <name>`, passing the key only through the environment.

## Repository layout

```
coastal-debris-segmentation/
├── README.md, README_EN.md
├── requirements.txt              training and local inference
├── requirements-data.txt         data collection
├── docs/method.md                final pipeline in detail (Korean)
├── src/                          synthesis, training, metric (newbase/ during the competition)
│   ├── synth.py                  synthesizer (the final models use recipes v4 and v26)
│   ├── train.py                  training (U-Net / frozen foundation model)
│   ├── valmix.py                 robustness validation set (six appearance groups)
│   ├── vfm_net.py                DINOv3 + Rein adapters + high-resolution decoder
│   ├── anyup/                    subset of AnyUp (CC BY 4.0)
│   ├── metric.py                 re-implementation of the competition metric
│   ├── postproc.py, chroma.py    decision rule, chroma normalization
│   ├── mine_hardneg.py           mining of real-background false positives
│   └── e2e_check.py, run_local.py   local test set and scoring, local notebook run
├── scripts/
│   ├── train_final.sh            training commands for the final models
│   ├── make_package.py           fills the submission package with weights
│   └── verifier/                 verifier data building and training
├── data/
│   ├── README.md                 expected layout, sources and licenses
│   └── scripts/                  background and AI-Hub collection/conversion
└── submission/                   final submission package (without weights)
    ├── predict.ipynb, requirements.txt
    └── assets/                   config.json, license notices, code/anyup, model/README.md
```

- `submission/predict.ipynb` and `assets/config.json` are the exact files I submitted.
- `src/synth.py` and `src/train.py` still contain experiment recipes not used in the final model (v5, v6, v17, v27, v28). I left them so the code that trained the final models stays unchanged.
- Paths such as `docs/…` and `research/…` in code comments refer to design notes in my competition workspace; they are not in this repository.
- `scripts/train_final.sh`, `scripts/make_package.py` and `scripts/verifier/export_dinov2.py` were written while preparing this repository. The first two gather run commands and package assembly steps that were spread across many scripts during the competition; the last re-creates backbone export code that was not kept (its output matches the submitted file tensor for tensor).

## Data and weights

This repository contains no imagery, datasets or model weights.

- **Background imagery**: five open orthoimagery sources (NAIP, ERI, LINZ, GSI, IGN). Scene counts and licenses per source are in [data/README.md](data/README.md). All allow commercial use and derivative works; the obligation is essentially attribution.
- **AI-Hub data** (71363, 506): used only to train the verifier. Redistribution of the originals is not allowed, so nothing is included. This work (the verifier) used AI-Hub data ("Coastal pollutant data", "AI-based national park change detection monitoring platform") built under the AI training data project of the National Information Society Agency (NIA), Korea.
- **Competition materials**: the organizer's baseline package and copies of the competition pages are not included.
- **Weights**: the 7 weight files of the final package (about 2.6 GB) are not included. The file list and sizes are in [submission/assets/model/README.md](submission/assets/model/README.md).
- **Pretrained weights**: torchvision ResNet-34 ImageNet (BSD-3-Clause), DINOv3 ViT-L/16 SAT-493M (DINOv3 License), DINOv2 ViT-B/14 reg4 (Apache-2.0), AnyUp (CC BY 4.0). Built with DINOv3.

## License

There is no license file for this repository yet. Included third-party code and notices keep their own licenses: `src/anyup/` and `submission/assets/code/anyup/` (AnyUp, CC BY 4.0), and `submission/assets/LICENSE_DINOv3.txt` (copy of the DINOv3 License).
