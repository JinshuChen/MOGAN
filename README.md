# MOGAN

### Official pytorch implementation of the paper: "MOGAN: Morphologic-structure-aware Generative Learning from a Single Image"

![avatar](https://github.com/JinshuChen/MOGAN/blob/main/fig1.png)

## Random image generation for ROI-based tasks

Given ROIs, MOGAN is able to generate samples with diverse appereances and reasonable structures.

## Link

[Arxiv](https://arxiv.org/abs/2103.02997)

## Usage

### Main Dependencies

```
matplotlib
scikit-image
scikit-learn
scipy
numpy
torch
torchvision
```

### Training

```
python run.py --mode f --config_file test.config
```

### TODO list:
- [x] Clean useless comments.
- [x] PEP8.
- [ ] Add full applications.
- [ ] Add the code for image fusing.
- [ ] Add more detailed instructions.

### Note that
The code is incomplete for now. But the code of model's structure and training process is avaliable. Experiments haven't been stoped on a more reliable structure design so the code in the repo shouldn't be the final version.