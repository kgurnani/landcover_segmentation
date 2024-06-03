# Mamba-based Semantic Segmentation on Remote Sensing Imagery

## Abstract

TODO

## Reproduce results

### Environment setup

- Install anaconda from [here](https://www.anaconda.com/download).
- Setup environment with:
```
conda create --no-default-packages -n mrp python=3.10
conda activate mrp
pip install requirements.txt
```

### Project setup
```
Invoke-WebRequest https://landcover.ai.linuxpolska.com/download/landcover.ai.v1.zip -OutFile landcover.ai.v1.zip
Expand-Archive landcover.ai.v1.zip
cd landcover.ai.v1
python split.py
cd ..
python setup_dataset.py
```

### Checkout jupyter notebooks

TODO