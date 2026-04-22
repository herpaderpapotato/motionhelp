# Motion Help

![Great wavy lines](docs/wavy.png)

It makes these. It helps with motion.
There's no clever filtering or peak detection at play here yet.
This one feels like it's got some real potential, and I don't hate the architecture.

Looking at that plot, it might look like noise, but the wiggle at the 23 second mark is spot on.

## Overview

1. You use this to build a dataset from your existing content.
2. And then you train a model on the data.
3. And that helps you build an even bigger dataset.
4. And that lets you train a version that really nails the tempo and magnitude on 95% of sequences, especially on the longer and more tedious scripts which frees you to focus on the bigger questions like, *should I be basing the pos on the hand, or the mouth in this scene?*

---

## Installation

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html), then:

```bash
git clone https://github.com/herpaderpapotato/motionhelp
cd motionhelp

conda create -p thefullpathtomotionhelp\.conda python=3.13
conda activate thefullpathtomotionhelp\.conda
conda install -c conda-forge "torchcodec=*=*cuda*" "torchvision=*=*cuda*" "torchaudio=*=*cuda*" -y

pip install opencv-python matplotlib tqdm PyYAML pillow scipy psutil polars ipykernel tensorboard requests onnx onnxscript
pip install ultralytics ultralytics-thop --no-deps
```

> **Note:** If you run into issues with missing system libraries, try:
> ```bash
> conda install libglib gdk-pixbuf ffmpeg -c conda-forge --force-reinstall
> ```

---

## Preparation

Preprocessed clips are sourced from your xbvr library. Currently supported source:

- **xbvr on MySQL** ✓
- Target a folder with matching filenames *(not yet)*
- From `main.db` *(not yet)*

Extract clips and labels from your video sources:

```bash
python .\scripts\prepare_videos.py
```

This will extract 5 segments from each of 50 video sources into `.\data\preprocessed` by default. Run with `--help` to see how to change those limits.

While that's running, open a second activated conda environment and start the spatial feature extractor in watch mode:

```bash
python .\scripts\extract_spatial.py --watch
```

You can leave it running or kill it once both scripts complete.

### Curate the dataset

```bash
python .\scripts\visualize_data.py
```

Filter to **Pending**. Press **A** to approve a clip or **D** to reject it.

Review carefully — the interpolation isn't always reliable and there are occasional timing issues with the encoded clips that I haven't fully tracked down yet. Manual review catches a lot of that.

---

## Initial Train

Once you have around **120 approved clips**, build the train/validation split:

```bash
python scripts\build_splits.py
```

Then train:

```bash
python.exe scripts/train_disposition.py
```

> `--batch-size 128` uses around 24 GB of VRAM. The default batch size of 16 uses about 2 GB.

It'll produce something. Not a very good something, but something — and it's enough to start using MSE to guide curation.

Evaluate your approved and pending clips:

```bash
python .\scripts\evaluate_scenes.py
```

Then open the visualizer again:

```bash
python .\scripts\visualize_data.py
```

Sort by **MSE**. Clips with high MSE are where predictions diverge most from labels — review those approved/pending clips and reject the ones that look wrong. Bad clips do sneak through manual review sometimes.

---

## Dataset Expansion and Curation

Pull more samples:

```bash
python .\scripts\prepare_videos.py --max-scenes 2000 --segments-per-video 10
```

While that's running, start these in separate activated conda environments:

```bash
python .\scripts\extract_spatial.py --watch
python .\scripts\evaluate_scenes.py --watch
```

Leave them running or kill them once `prepare_videos.py` finishes.

Then curate again:

```bash
python .\scripts\visualize_data.py
```

Sort by MSE. For low-MSE pending clips you can just hold **A** to approve in bulk. Mid-to-high MSE clips need a closer look before approving.

---

## Train N+1

Once you have around **1100–1200 approved clips**, rebuild the split and retrain:

```bash
python scripts\build_splits.py
python.exe scripts/train_disposition.py
```

To monitor training graphs while it runs, open another activated conda environment and run:

```bash
tensorboard --logdir runs
```

Then open [http://127.0.0.1:6006](http://127.0.0.1:6006) in your browser.

After 5–10 epochs it'll typically hit its best output and start overtraining. The best checkpoint is saved to `data/models/checkpoints_disposition/best_disposition.pt`.

---

## Generation

To generate a funscript prediction for a video:

```bash
python.exe scripts/predict_disposition.py --vr --video benchmark_8k.mp4 --out benchmark_8k.funscript
```

Useful flags:

| Flag | Description |
|---|---|
| `--vr` / `--no-vr` | 180° SBS VR input or standard 2D video |
| `--playback` | Watch the prediction in a GUI as it runs (slower) |
| `--start-time` | Start offset in seconds |
| `--duration` | Duration to process in seconds |

---

## Tune

Once results are okay — and quite often they're quite good — but you want better, refresh your MSE scores:

```bash
python .\scripts\evaluate_scenes.py --overwrite
```

Then review high-MSE approved clips again:

```bash
python .\scripts\visualize_data.py
```

Reject anything that doesn't look right, then go back to [Train N+1](#train-n1) and repeat.

---

## FAQ

**Q1: Why do I have to train my own model — why can't I have yours?**  
A: The scripts I've trained on, I don't have expressed permission to distribute a model trained from. Ethically I'd need that first, which would be a significant undertaking and not really my forte.

**Q2: Conda error, CUDA error, some other error — broken!**  
A: This is a pretty niche thing to be running. The answer is either that you know how to fix your error, or you know how to use AI to help you fix your error. Happy to accept any doc improvements that make sense.

**Q3: Your model is bad and your code is bad.**  
A: Yeah, it's all gone through a lot of changes and iterations. At some point you have to put things out there though. I'm still not 100% sure if I need all 3 spatial features, or if the finetuned YOLO is actually necessary (swapping it would mean regenerating all spatial data), or if the DDL setting is actually better, or if the N-size YOLO models would be better, or a different feature extractor entirely, etc, etc, etc.

**Q4: You're not really doing any augmentation in your training loop.**
A: Yeah. I'll get to it, but I'm still mulling over Q3 stuff.

**Q5: Wow that's one AI ass README.MD you had AI generated!**
A: Kinda, check the previous version for the hard to follow notes I had here previously.

**Q6: This whole thing is AI vibe coded slop.**
A: Some of the newer models have definitely been an accelerator for me in this space, although I'd point to my wall of text posts elsewhere as an evidence that this is notably better than the jank python I normally write! 