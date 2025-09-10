This repository provides the full workflow for nematode trait analysis, combining BoxInst (Detectron2 + AdelaiDet) for detection and Nema-SAM for instance segmentation and trait extraction.

⚠️ Note
It is strongly recommended to create two separate Python environments: one for BoxInst (Detectron2 + AdelaiDet) and one for Nema-SAM, to avoid dependency conflicts.
Please install PyTorch and Detectron2 versions that are compatible with your local CUDA setup. For machines without GPU, a CPU version can be installed (with slower inference speed).

# Repository structure (excerpt)

```
.
├── BoxInst/
│   └── AdelaiDet-master/
│       ├── configs/
│       ├── adet/                # AdelaiDet source
│       ├── detectron2/          # Detectron2 (if cloned as submodule/source)
│       ├── tools/
│       ├── predict.ipynb        # Example notebook for inference
│       └── ...
├── Nema-SAM/
│   └── ...                      # Nema-SAM and trait analysis scripts/notebooks
└── README.md
```



# Requirements:
Please follow the 
1. First crop the analysed image input 12 part, then input the cropped image into the BoxInst model to detect the nematode.
   
