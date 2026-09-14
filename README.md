## Beyond the Survey: A Systematic Empirical Study of Detection and Association in Visual MOT

Official implementation repository for the paper **"Beyond the Survey: A Systematic Empirical Study of Detection and Association in Visual MOT"**, accepted for publication in *Artificial Intelligence Review*.

> **Note:** This repository also includes the official Python implementation for:  
> **"Adaptive Confidence Threshold for ByteTrack in Multi-Object Tracking"** (*ICCAIS 2023*, [arXiv 2312.01650](https://arxiv.org/abs/2312.01650)) at this folder [bytetrack](trackers/bytetrack).

---

### Overview

This repository provides tracking implementations for algorithms evaluated in our study: **SORT, DeepSORT, MOTDT, FairMOT, ByteTrack (AdaptByteTrack), OCSORT, DeepOCSORT, and VisualRFS**. We provide CMC (Camera Motion Compensation) for ByteTrack (AdaptByteTrack), VisualRFS, and OC-SORT. Evaluation Scores/Metrics are: 
- [CLEAR](https://link.springer.com/article/10.1155/2008/246309), [IDF1](https://arxiv.org/pdf/1609.01775), [HOTA](https://arxiv.org/abs/2009.07736) evaluation available at [JonathonLuiten/TrackEval](https://github.com/JonathonLuiten/TrackEval).
- Tracking Effort Measure [TEM](https://arxiv.org/abs/2212.08536) ([vpulab/MOT-evaluation](https://github.com/vpulab/MOT-evaluation)).

For other evaluated methods, please clone their respective repositories and follow the authors' original execution instructions.

---

### Datasets & Detection Outputs

#### Pre-extracted Detections can be downloaded from Hugging Face ([linhmv/VisualMOT](https://huggingface.co/datasets/linhmv/VisualMOT)).
We provide extracted detection files with confidence scores $[0, 1]$ in the ([`./dets/`](./dets/)) directory:

| Detector                       | Venue / Source | Paper / Link |
|:-------------------------------| :--- | :--- |
| POI: `detector_poi`            | ECCV 2016 | [arXiv:1610.06136](https://arxiv.org/pdf/1610.06136) |
| JDE: `detector_jde`            | ECCV 2020 | [arXiv:1909.12605](https://arxiv.org/abs/1909.12605) |
| TraDeS: `detector_trades`      | CVPR 2021 | [arXiv:2103.08808](https://arxiv.org/abs/2103.08808) |
| FairMOT: `detector_fairmot128` | IJCV 2021 | [arXiv:2004.01888](https://arxiv.org/abs/2004.01888) |
| GSDT: `detector_gsdt`          | ICRA 2021 | [arXiv:2006.13164](https://arxiv.org/abs/2006.13164) |
| CSTrack: `detector_cstrack`    | TIP 2022 | [arXiv:2010.12138](https://arxiv.org/abs/2010.12138) |
| YOLOX: `detector_bytetrack`    | ECCV 2022 | [arXiv:2110.06864](https://arxiv.org/abs/2110.06864) |
| YOLOv11: `detectors_yolov11`   | arXiv 2024 | [arXiv:2410.17725](https://arxiv.org/abs/2410.17725) |

  - POI: Relies on [ETHZ](https://ieeexplore.ieee.org/document/4587581), [Caltech Pedestrian](https://ieeexplore.ieee.org/abstract/document/5206631), and a self-collected surveillance dataset.
  - TraDeS: Utilizes a [CrowdHuman](https://arxiv.org/abs/1805.00123) pre-trained model for 2D tracking alongside the [MOTChallenge](https://motchallenge.net/) dataset.
  - JDE, FairMOT, CSTrack, GSDT: Fine-tuned on the "[Mix of Six](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md)" dataset, which combines [Caltech Pedestrian](https://ieeexplore.ieee.org/abstract/document/5206631), [CityPersons](https://arxiv.org/abs/1702.05693), [ETHZ](https://ieeexplore.ieee.org/document/4587581), [MOTChallenge](https://motchallenge.net/), [CUHK-SYSU](https://arxiv.org/abs/1604.01850), and [PRW](https://arxiv.org/abs/1604.02531).
  - YOLOX: Trained on a combination of [MOTChallenge](https://motchallenge.net/), [CrowdHuman](https://arxiv.org/abs/1805.00123), [CityPersons](https://arxiv.org/abs/1702.05693), and [ETHZ](https://ieeexplore.ieee.org/document/4587581).
  - YOLOv11: `YOLO11x`, only trained on [COCO Detection](https://cocodataset.org/#overview) dataset
  
  

#### Evaluation Datasets
Dataset Name | Year                                                                                         | Source|
| :--- |:---------------------------------------------------------------------------------------------| :--- |
| [MOTChallenge](https://motchallenge.net/)| [2016](https://arxiv.org/abs/1603.00831), [2017](https://motchallenge.net/data/MOT17/), [2020](https://arxiv.org/abs/2003.09003) | 
|[DanceTrack](https://arxiv.org/abs/2111.14690)| CVPR 2022                                                                                    |[DanceTrack/DanceTrack](https://github.com/DanceTrack/DanceTrack)
|[SportsMOT](https://arxiv.org/abs/2304.05170)| ICCV 2023                                                                                    |[MCG-NJU/SportsMOT](https://github.com/MCG-NJU/SportsMOT)
|[CrowdTrack](https://arxiv.org/abs/2507.02479)| arXiv 2025                                                                                   | [loseevaya/CrowdTrack](https://github.com/loseevaya/CrowdTrack)

---

### Evaluated Tracking Algorithms

#### 1. Analytical Data Association
*Hand-crafted motion & appearance models*

| Method                                                                                                                                                                             | Ref. Index |     Year     | Source Code |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| :---: |:------------:| :--- |
| **[SORT](https://arxiv.org/abs/1602.00763)**                                                                                                                                       | [23] | ICIP 2016 | [abewley/sort](https://github.com/abewley/sort) |
| **[DeepSORT](https://arxiv.org/abs/1703.07402)**                                                                                                                                   | [27] |  ICIP 2017   | [nwojke/deep_sort](https://github.com/nwojke/deep_sort) |
| **[MOTDT](https://arxiv.org/abs/1809.04427)**                                                                                                                                      | [28] |  ICME 2018   | [longcw/MOTDT](https://github.com/longcw/MOTDT) |
| **[FairMOT](https://arxiv.org/abs/2004.01888)**                                                                                                                                    | [29] |  IJCV 2021   | [ifzhang/FairMOT](https://github.com/ifzhang/FairMOT) |
| **[ByteTrack](https://arxiv.org/abs/2110.06864)**                                                                                                                                  | [24] |  ECCV 2022   | [FoundationVision/ByteTrack](https://github.com/FoundationVision/ByteTrack) |
| **[AdaptByteTrack](https://arxiv.org/abs/2312.01650)**                                                                                                                             | [75] | ICCAIS  2023 | [linh-gist/AdaptConfByteTrack](https://github.com/linh-gist/AdaptConfByteTrack) |
| **[OCSORT](https://arxiv.org/abs/2203.14360)**                                                                                                                                     | [25] |  CVPR 2023   | [noahcao/OC_SORT](https://github.com/noahcao/OC_SORT) |
| **[DeepOCSORT](https://arxiv.org/abs/2302.11813)**                                                                                                                                 | [30] |  ICIP 2023   | [gerardmaggiolino/deep-oc-sort](https://github.com/gerardmaggiolino/deep-oc-sort) |
| **[StrongSORT](https://arxiv.org/abs/2202.13514)**                                                                                                                                 | [32] |   TMM 2023   | [dyhBUPT/StrongSORT](https://github.com/dyhBUPT/StrongSORT) |
| **[VisualRFS](https://arxiv.org/abs/2407.08872)**                                                                                                                                  | [1] |   PR 2024    | [linh-gist/VisualRFS](https://github.com/linh-gist/VisualRFS) |
| **[HybridSORT](https://arxiv.org/abs/2308.00783)**                                                                                                                                 | [31] |  AAAI 2024   | [ymzis69/HybridSORT](https://github.com/ymzis69/HybridSORT) |
| **[TrackTrack](https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html)**                                | [26] |  CVPR 2025   | [kamkyu94/TrackTrack](https://github.com/kamkyu94/TrackTrack) |

#### 2. Deep Learning Data Association
*Learned feature-based association*

| Method                                                                                               | Ref. Index |    Year     | Source Code |
|:-----------------------------------------------------------------------------------------------------| :---: |:-----------:| :--- |
| **[SUSHI](https://arxiv.org/abs/2212.03038)**                                                        | [33] | CVPR 2023 | [dvl-tum/sushi](https://github.com/dvl-tum/sushi) |
| **[LTTrack](https://ieeexplore.ieee.org/abstract/document/10536914)**                                | [34] | TCSVT 2024  | [linjiaping1/LTTrack](https://github.com/linjiaping1/LTTrack) |
| **[LG-MOT](https://arxiv.org/abs/2406.04844)**                                                       | [35] | TCSVT 2025  | [weslee88524/lg-mot](https://github.com/weslee88524/lg-mot) |

#### 3. End-to-End (E2E) Data Association
*Joint detection and association learning*

| Method                                                                        | Ref. Index |    Year    | Source Code |
|:------------------------------------------------------------------------------| :---: |:----------:| :--- |
| **[MOTR](https://arxiv.org/pdf/2105.03247)**                                  | [17] | ECCV 2022  | [megvii-research/MOTR](https://github.com/megvii-research/MOTR) |
| **[MeMOTR](https://arxiv.org/abs/2307.15700)**                                | [41] | ICCV 2023  | [mcg-nju/memotr](https://github.com/mcg-nju/memotr) |
| **[MOTIP](https://arxiv.org/abs/2403.16848)**                                 | [42] | CVPR 2025  | [MCG-NJU/MOTIP](https://github.com/MCG-NJU/MOTIP) |
| **[CO-MOT](https://arxiv.org/abs/2305.12724)**                                | [43] | ICLR 2025  | [BingfengYan/CO-MOT](https://github.com/BingfengYan/CO-MOT) |
| **[SambaMOTR](https://arxiv.org/abs/2410.01806)**                             | [39] | ICLR 2025  | [mattiasegu/sambamotr](https://github.com/mattiasegu/sambamotr) |





### Usage
1. **Set Up Python Environment**
    - Create a `conda` Python environment and activate it:
        ```sh
        conda create --name virtualenv python==3.8.0
        conda activate virtualenv
        ```
    - lone this repository recursively to have pybind11
        ```sh
        git clone --recursive https://github.com/linh-gist/VisualMOT.git
        ```
    - Install Packages
        ```sh
        numpy==1.23.1
        opencv-python==4.9.0.80
        loguru==0.7.2
        scipy==1.10.1
        lap==0.5.12
        cython_bbox==0.1.5
        matplotlib==3.5.3
        filterpy==1.4.5
        motmetrics==1.4.0
        openpyxl==3.1.5
        pycocotools==2.0.7
        tabulate==0.9.0
        # git clone https://github.com/JonathonLuiten/TrackEval.git
        # cd TrackEval, python setup.py build develop
        ```

2. **Prepare Data**
    - Datasets: 
        - MOT16, MOT17, MOT20, DanceTrack, SportsMOT, CrowdTrack
        - You can also run with your custom dataset but need a detector

3. **Run the Tracking Demo**
   - Change parameters in `make_parser()` in `track.py` such as `use_gmc`, `data_dir` (MOTChallenge GT data)
   - Run `python track.py`


### Citation
If you find this project useful in your research, please consider citing by:

```
@article{van2026beyond,
  title={Beyond the Survey: A Systematic Empirical Study of Detection and Association in Visual MOT},
  author={Linh Van Ma and Juhua Hu and Wei Cheng and Unse Fatima and Moongu Jeon},
  booktitle={Artificial Intelligence Review},
  year={2026},
  publisher={Springer}
}
@inproceedings{van2023adaptive,
  title={Adaptive Confidence Threshold for ByteTrack in Multi-Object Tracking},
  author={Linh Van Ma and Muhammad Ishfaq Hussain and JongHyun Park and Jeongbae Kim and Moongu Jeon},
  booktitle={2023 12th International Conference on Control, Automation and Information Sciences (ICCAIS)},
  pages={370--374},
  year={2023},
  organization={IEEE}
}
```

### Acknowledgement
A part of the code is borrowed from [SORT](https://github.com/abewley/sort), [DeepSORT](https://github.com/nwojke/deep_sort), [MOTDT](https://arxiv.org/abs/1809.04427), [FairMOT](https://arxiv.org/abs/2004.01888), [ByteTrack](https://arxiv.org/abs/2110.06864), [OCSORT](https://arxiv.org/abs/2203.14360), [DeepOCSORT](https://arxiv.org/abs/2302.11813), and [VisualRFS](https://arxiv.org/abs/2407.08872). Thanks for their wonderful works.
