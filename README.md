This is the official implementation of the paper [STEP: Spatial Temporal Graph Convolutional Networks for Emotion Perception from Gaits](https://aaai.org/ojs/index.php/AAAI/article/view/5490). Please add the following citation in your work if you use our code:

``@inproceedings{bhattacharya2020step,
author = {Bhattacharya, Uttaran and Mittal, Trisha and Chandra, Rohan and Randhavane, Tanmay and Bera, Aniket and Manocha, Dinesh},
title = {STEP: Spatial Temporal Graph Convolutional Networks for Emotion Perception from Gaits},
year = {2020},
publisher = {AAAI Press},
booktitle = {Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence},
pages = {1342–1350},
numpages = {9},
series = {AAAI’20}
}``

We have also released the Emotion-Gait dataset with this code, which is available for download here: [https://go.umd.edu/emotion-gait](https://go.umd.edu/emotion-gait).

1. generator_cvae is the generator.
2. classifier_stgcn_real_only is the baseline classifier using only the real 342 gaits.
3. classifier_stgcn_real_and_synth is the baseline classifier using both real 342 and N synthetic gaits.
4. clasifier_hybrid is the hybrid classifier using both deep and physiologically-motivated features.
