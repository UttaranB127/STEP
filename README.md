This is the official implementation of the paper [STEP: Spatial Temporal Graph Convolutional Networks for Emotion Perception from Gaits](https://obj.umiacs.umd.edu/gamma-umd-website-imgs/pdfs/affectivecomputing/STEP.pdf). Please add the following citation in your work if you use our code:

``@inproceedings{bhattacharya2020step,

  title={STEP: Spatial Temporal Graph Convolutional Networks for Emotion Perception from Gaits.},
  
  author={Bhattacharya, Uttaran and Mittal, Trisha and Chandra, Rohan and Randhavane, Tanmay and Bera, Aniket and Manocha, Dinesh},
  
  booktitle={AAAI},
  
  pages={1342--1350},
  
  year={2020}
}``

We have also released the Emotion-Gait dataset with this code, which is available for download [here](https://go.umd.edu/emotion-gait).

1. generator_cvae is the generator.
2. classifier_stgcn_real_only is the baseline classifier using only the real 342 gaits.
3. classifier_stgcn_real_and_synth is the baseline classifier using both real 342 and N synthetic gaits.
4. clasifier_hybrid is the hybrid classifier using both deep and physiologically-motivated features.
