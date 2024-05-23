# Mamba-based Semantic Segmentation on Remote Sensing Imagery

- install miniconda
- conda create --no-default-packages -n mrp python=3.10
- conda activate mrp
- pip install requirements.txt
- Invoke-WebRequest https://landcover.ai.linuxpolska.com/download/landcover.ai.v1.zip -OutFile landcover.ai.v1.zip
- Expand-Archive landcover.ai.v1.zip
- cd landcover.ai.v1
- python split.py
- cd ..