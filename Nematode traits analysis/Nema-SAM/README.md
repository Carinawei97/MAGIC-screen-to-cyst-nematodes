# Nema-SAM

This folder contains the **Nema-SAM** model and scripts for fine-grained nematode instance segmentation and confidence-based prediction.

---

## 1. Environment Setup

You can set up the environment in multiple ways:

### A. Using `cog.yaml`
If you are deploying with [Cog](https://github.com/replicate/cog), simply run:
```bash
cog run python predict.py
```
Cog will automatically create the environment from `cog.yaml`.

### B. Using `environment.yml` (recommended for Conda)
```bash
conda env create -f environment.yml
conda activate nemasam
```

### C. Using `requirements.txt` (for pip/virtualenv)
```bash
python -m venv nemasam_env
source nemasam_env/bin/activate   # (Linux/Mac)
nemasam_env\Scripts\activate    # (Windows)

pip install -r requirements.txt
```

---

## 2. Model Weights

The Nema-SAM model is provided in the **[Releases](https://github.com/Carinawei97/MAGIC-screen-to-cyst-nematodes/releases)** section of this repository.  

1. Go to the [Releases page](https://github.com/Carinawei97/MAGIC-screen-to-cyst-nematodes/releases).  
2. Download the file named **`Nema-SAM`**.  
3. Place it in the appropriate directory (update the path in `predict.ipynb`).  

---

## 3. Prediction Workflow

The notebook **`predict.ipynb`** demonstrates how to run Nema-SAM for nematode segmentation and trait extraction.

### Input
- A **single cropped nematode image** detected by **BoxInst**.

### Output
- A **refined instance mask** of the nematode.  
- A **confidence score** indicating prediction certainty.  
- Visualization and optional saving of the segmented nematode.  

### Usage
1. Open the notebook:
   ```bash
   jupyter notebook predict.ipynb
   ```
2. Modify the following paths inside the notebook:
   - `MODEL_PATH`: path to the downloaded Nema-SAM weights.  
   - `INPUT_PATH`: path to the cropped nematode image (from BoxInst).  
3. Run all cells step by step.  
4. The output will include the segmented mask and associated confidence.  

---

## 4. Batch Prediction Workflow

The notebook **`batch_predict.ipynb`** demonstrates how to run Nema-SAM for nematode segmentation and trait extraction.

### Input

## Notes
- Ensure BoxInst detection is run **before** Nema-SAM (input images should already be cropped around individual nematodes).  









