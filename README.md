# A HYBRID TRANSFORMER-MAMBA APPROACH FOR LANDCOVER AERIAL IMAGERY SEGMENTATION

## Abstract

Aerial imagery segmentation is crucial for various applications, but traditional methods face challenges in processing efficiency and accuracy. This study proposes a novel hybrid Transformer-Mamba approach for semantic segmentation of landcover aerial imagery, aiming to improve performance and computational efficiency. We evaluated three model configurations using the Landcover dataset: a Transformer-only baseline and two hybrid Transformer-Mamba models. Performance was assessed using standard metrics and throughput. The hybrid model with the highest Mamba-to-Transformer ratio showed slight improvements across all metrics compared to the baseline. It achieved the highest IoU (0.57) and throughput (9.91 images/second), a 10.5% speed increase over the baseline. The hybrid approach shows promise for enhancing aerial imagery segmentation, offering modest improvements in accuracy and efficiency. Further research is needed to address class imbalance and optimize the architecture.

## Reproduce results

Run each of the model variants (12t-0m, 2t-5m, 1t-1m) in Google Colab.
