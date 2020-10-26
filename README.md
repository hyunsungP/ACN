# ACN
Attentional Combination Network

Face alignment with GUI

You can download weights for ACN from the link below:
https://drive.google.com/file/d/12aFpNTJIJXIbfM6caESP_XlHLGGezVDY/view?usp=sharing
Put ACN.tar into ./weights/ACN.tar 

You can download weights for pyramidbox from the link below:
https://github.com/cs-giung/face-detection-pytorch
Put pyramidbox_120000_99.02.pts into ./detectors/pyramidbox/weights/pyramidbox_120000_99.02.pth

Run demo.py

Components
- Face detection: pyramidbox, https://github.com/cs-giung/face-detection-pytorch
- Attentnion mask generation: CPN, https://github.com/GengDavid/pytorch-cpn
- Heatmap regression: DU-Net, https://github.com/zhiqiangdon/CU-Net
  
