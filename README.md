# Leveraging Front and Side Cues for Monocular 3D Object Detection

## Introduction

<!-- [ALGORITHM] -->

FSG, also can be regarded as PGD-FCOS3D++, is a simple yet effective monocular 3D detector. It enhances the FCOS3D baseline by leveraging front and side cues and improving instance depth estimation.

## Results

### KITTI

Detailed performance on KITTI 3D detection (3D/BEV) is as follows, evaluated by AP11 and AP40 metric:

|             |     Easy      |    Moderate    |     Hard      | mAP_11/mAP_40 |
|-------------|:-------------:|:--------------:|:-------------:|:-------------:|
| Car (AP11)  | 24.14 / 32.31 | 19.60 / 25.59  | 17.63 / 22.15 |[model](https://drive.google.com/file/d/1x7M97oUWhgXxBWwpv1I7weT-0KZYQyyq/view?usp=sharing)|
| Car (AP40)  | 19.07 / 28.56 | 14.31 / 21.16  | 11.68 / 17.48 ||

Note: mAP represents Car moderate 3D strict AP11 / AP40 results. Because of the limited data for pedestrians and cyclists, the detection performance for these two classes is usually unstable. Therefore, we only list car detection results here. In addition, AP40 is a more recommended metric for reference due to its much better stability.
