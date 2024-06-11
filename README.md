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
python dataset_split.py
python setup_dataset.py
python img_augmentations.py
```

### Checkout jupyter notebooks

TODO

## Citations

```
@InProceedings{Boguszewski_2021_CVPR,
    author = {Boguszewski, Adrian and Batorski, Dominik and Ziemba-Jankowska, Natalia and Dziedzic, Tomasz and Zambrzycka, Anna},
    title = {LandCover.ai: Dataset for Automatic Mapping of Buildings, Woodlands, Water and Roads from Aerial Imagery},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month = {June},
    year = {2021},
    pages = {1102-1110}
}
```

```
@misc{zhao2024rsmamba,
      title={RS-Mamba for Large Remote Sensing Image Dense Prediction}, 
      author={Sijie Zhao and Hao Chen and Xueliang Zhang and Pengfeng Xiao and Lei Bai and Wanli Ouyang},
      year={2024},
      eprint={2404.02668},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```
@article{huang2024localmamba,
  title={LocalMamba: Visual State Space Model with Windowed Selective Scan},
  author={Huang, Tao and Pei, Xiaohuan and You, Shan and Wang, Fei and Qian, Chen and Xu, Chang},
  journal={arXiv preprint arXiv:2403.09338},
  year={2024}
}
```