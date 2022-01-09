# Face Tells Detailed Expression

## Overview
This repository provides an implementation of paper, [Face Tells Detailed Expression: Generating Comprehensive Facial Expression Sentence through Facial Action Units](https://link.springer.com/chapter/10.1007/978-3-030-37734-2_9).

## Prerequisites
- `Python 2.7` (code has been tested with this version)
- Download [vgg_face_weights](https://drive.google.com/file/d/1OCdnN8NJsLSiDYycZHF3BeL1pvAm8-4V/view?usp=sharing) and put it in the project.
- Download [stanford-corenlp-3.4.1]() and put it in [here]().

## Datasets
- Text-based dataset with comprehensive facial expression sentence is available [here](https://github.com/joannahong/Face-Tells-Detailed-Expression-Dataset).
- We provide the code that is executable in CK+ dataset.
- We have cooked the dataset based on the format below. We provide several examples in [here]().
### CK+ dataset
```
data_root 
├── CK+ (or any other speaker-specific folder)
|	├── cohn-kanade-images/ (will contain the aligned video image frames)
|	├── FACS/		(will contain the facial action units info.)
|	├── Emotion/	(will contain emotion info.) 
```

## Citation
Please cite the following paper if you have use this code:
```
@inproceedings{hong2020face,
  title={Face Tells Detailed Expression: Generating Comprehensive Facial Expression Sentence Through Facial Action Units},
  author={Hong, Joanna and Lee, Hong Joo and Kim, Yelin and Ro, Yong Man},
  booktitle={International Conference on Multimedia Modeling},
  pages={100--111},
  year={2020},
  organization={Springer}
}
