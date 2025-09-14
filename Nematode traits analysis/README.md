# Nematode Traits Analysis

This repository provides the full workflow for nematode trait analysis, combining **BoxInst (Detectron2 + AdelaiDet)** for detection and **Nema-SAM** for instance segmentation and trait extraction.

> ⚠️ **Note**  
> - It is strongly recommended to create **two separate Python environments**: one for **BoxInst (Detectron2 + AdelaiDet)** and one for **Nema-SAM**, to avoid dependency conflicts.  
> - Please install PyTorch and Detectron2 versions that are compatible with your local CUDA setup. For machines without GPU, a CPU version can be installed (with slower inference speed).

---

## Repository structure (excerpt)

```
.
├── BoxInst/
│   └── AdelaiDet-master/
│       ├── configs/
│       ├── adet/                # AdelaiDet source
│       ├── detectron2/          # Detectron2
│       ├── tools/
│       ├── predict.ipynb        # Example notebook for inference
│       └── ...
├── Nema-SAM/
│   └── ...                      # Nema-SAM and trait analysis scripts/notebooks
└── README.md
```

---

## Requirements

### A. BoxInst environment (Detectron2 + AdelaiDet)

**Recommended: create a new Conda environment (Python 3.8/3.9, CUDA 11.8 example)**

```bash
# 1) Create and activate environment
conda create -n boxinst_env python=3.9 -y
conda activate boxinst_env

# 2) Install PyTorch (choose the right CUDA or CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3) Common dependencies
pip install jupyter matplotlib tqdm opencv-python pyyaml cython

# 4) Install Detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# (alternative: clone and install locally)
# git clone https://github.com/facebookresearch/detectron2.git
# cd detectron2 && pip install -e . && cd ..

# 5) Install AdelaiDet
cd BoxInst/AdelaiDet-master
pip install -e .

# 6) Additional dependencies (if not automatically installed)
pip install termcolor tabulate yacs iopath fvcore==0.1.5.post20221221

# 7) Test
python -c "import torch, detectron2; print('Torch:', torch.__version__); print('Detectron2 OK')"
```

> **Windows note**: You may need Microsoft C++ Build Tools if compilation errors occur. Alternatively, use WSL/Ubuntu for smoother installation.

---

### B. Nema-SAM environment

Create a **separate environment** for Nema-SAM to prevent conflicts.

```bash
# 1) Create and activate environment
conda create -n nemasam_env python=3.8 -y
conda activate nemasam_env

# 2) Install PyTorch (CUDA or CPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3) Common dependencies
pip install opencv-python scikit-image scipy numpy matplotlib tqdm jupyter

```

Their is a `requirements.txt` file is provided under the Nema-SAM folder, install with:

```bash
pip install -r requirements.txt
```

---

## Prediction

To run nematode detection with **BoxInst**:

```bash
# 1) Activate BoxInst environment
conda activate boxinst_env

# 2) Enter AdelaiDet directory
cd BoxInst/AdelaiDet-master

# 3) Start Jupyter
jupyter notebook

# 4）Download the model of BoxInst
Please through Releases to get the model for nematode detection

# 5) Get the test image
Through the 'Test' folder to get the test image
```

Open and run **`/BoxInst/AdelaiDet-master/predict.ipynb`** in the browser:

- **Config file**: set the YAML config (`/BoxInst/AdelaiDet-master/training_dir/config.yaml`).  
- **Weights**: specify the trained `.pth` model file in `MODEL.WEIGHTS`. The weight is in the Releases.
- **Input/Output**: define input images or directories and output folder paths.

To run nematode instance mask refine with **Nema-SAM**:
```bash
# 1) Activate Nema-SAM environment
conda activate nemasam_env

# 2) Enter Nema-SAM directory
cd Nema-SAM

# 3) Start Jupyter
jupyter notebook
```

Open and run **`predict.ipynb`** in the browser:

- **Weights**: specify the trained `.pth` model file in `MODEL.WEIGHTS`. The weight is in the Releases.
- **Input/Output**: define input images or directories and output folder paths.

---

## Run a test image

<div align="center">
    <img src="Test/1;1;1;23;510;01/1;1;1;23;510;01_2022-05-12_08_51_42.527516_Lilo.png" width="800" >
</div>

1) Crop the image into 12 parts



## FAQ

- **Detectron2 install fails?**  
  Ensure PyTorch/CUDA versions match. If needed, install CPU-only version or use Linux/WSL.  

- **AdelaiDet `pip install -e .` fails?**  
  Usually a compiler issue. Install MSVC Build Tools on Windows, or switch to Linux/WSL.  

- **Empty results or visualization errors?**  
  Verify config file and weight file consistency, and check image preprocessing.  

---

## Citation

If you use this repository in your research, please cite the relevant works (BoxInst, Detectron2, AdelaiDet, SAM):

```bibtex
@inproceedings{tian2020boxinst,
  title={BoxInst: High-Performance Instance Segmentation with Box Annotations},
  author={Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  booktitle={CVPR},
  year={2020}
}

@misc{wu2019detectron2,
  title={Detectron2},
  author={Yuxin Wu and Alexander Kirillov and others},
  year={2019},
  howpublished={https://github.com/facebookresearch/detectron2}
}

@misc{adelaiDet,
  title={AdelaiDet},
  howpublished={https://github.com/aim-uofa/AdelaiDet}
}

@article{Zhao2023segment,
  title={Fast Segment Anything},
  author={Xu Zhao, Wenchao Ding and others},
  year={2023}
}
```

---


