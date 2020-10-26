# ACN
Attentional Combination Network

Face alignment with GUI

You can download weights file from the linke below:
https://drive.google.com/file/d/12aFpNTJIJXIbfM6caESP_XlHLGGezVDY/view?usp=sharing

Put ACN.tar into ./weights/ACN.tar and run demo.py

Components
- Face detection: pyramidbox
  https://github.com/cs-giung/face-detection-pytorch
- Attentnion mask generation: CPN
  https://github.com/GengDavid/pytorch-cpn
- Heatmap regression: DU-NET
  https://github.com/zhiqiangdon/CU-Net
  
