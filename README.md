# Pixel Consensus Voting for Panoptic Segmentation (CVPR 2020)

<div align="center">
  <img width="500" alt="backproj" src="https://user-images.githubusercontent.com/17956191/119717745-b473d900-be34-11eb-81f1-16e7a6bcfd96.png">
  <img alt="panel" src="https://user-images.githubusercontent.com/17956191/119717968-ff8dec00-be34-11eb-86c6-c81f0ec8dae0.png">
</div>


The core of our approach, Pixel Consensus Voting, is a framework for instance segmentation based on the Generalized Hough transform. Pixels cast discretized, probabilistic votes for the likely regions that contain instance centroids. At the detected peaks that emerge in the voting heatmap, backprojection is applied to collect pixels and produce instance masks. Unlike a sliding window detector that densely enumerates object proposals, our method detects instances as a result of the consensus among pixel-wise votes. We implement vote aggregation and backprojection using native operators of a convolutional neural network. The discretization of centroid voting reduces the training of instance segmentation to pixel labeling, analogous and complementary to FCN-style semantic segmentation, leading to an efficient and unified architecture that jointly models things and stuff. We demonstrate the effectiveness of our pipeline on COCO and Cityscapes Panoptic Segmentation and obtain competitive results. 

## Quick Intro
- The codebase contains the essential ingredients of PCV, including various spatial discretization schemes and convolutional backprojection inference. The network backbone is a simple FPN on ResNet.
- Visualzier 1 ([vis.py](src/vis.py)): loads a single image into a dynamic, interacive interface that allows users to click on pixels to inspect model prediction. It is built on matplotlib interactive API and jupyter widgets. Under the hood it's React.
- Visualizer 2 ([pan_vis.py](src/pan_vis.py)): A global inspector that take panoptic segmentation prediction and displays prediction segments against ground truth. Useful to track down which images make the most serious error and how.

- The core of PCV is contained in [src/pcv](src/pcv). The results reported in the paper uses [src/pcv/pcv_basic](src/pcv/pcv_basic.py). There are also a few modification ideas that didn't work out e.g. "inner grid collapse" ([src/pcv/pcv_igc](src/pcv/pcv_igc.py)), erasing boundary loss [src/pcv/pcv_boundless](src/pcv/pcv_boundless.py), smoothened gt assignment [src/pcv/pcv_smooth](src/pcv/pcv_smooth.py).
- The deconv voting filter weight intializaiton is in [src/pcv/components/ballot.py](src/pcv/components/ballot.py). Different deconv discretization schemes can be found in [src/pcv/components/grid_specs.py](src/pcv/components/grid_specs.py). [src/pcv/components/snake.py](src/pcv/components/snake.py) manages the generation of snake grid on which pcv operates.

- The backprojection code is in [src/pcv/inference/mask_from_vote.py](src/pcv/inference/mask_from_vote.py). Since this is a non-standard procedure of convolving a filter to do equality comparison, I implemented a simple conv using advanced indexing. See the function [src/pcv/inference/mask_from_vote.py:unroll_img_inds](src/pcv/inference/mask_from_vote.py#L110-L119). For a fun side-project, I am thinking about rewriting the backprojection in Julia and generate GPU code (ptx) directly through LLVM. That way we don't have to deal with CUDA kernels that are hard to maintain. 
- The main entry point is [run.py](run.py) and [src/entry.py](src/entry.py)


## Getting Started

### Dependencies
- pytorch==1.4.0
- fabric (personal toolkit that needs to be re-factored in)
- pycocotools

~~~python
python run.py -c train
python run.py -c evaluate
~~~
runs the default PCV configuration reported in Table 3 of the paper. 


## Bibtex

```bibtex
@inproceedings{pcv2020,
  title={Pixel consensus voting for panoptic segmentation},
  author={Wang, Haochen and Luo, Ruotian and Maire, Michael and Shakhnarovich, Greg},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9464--9473},
  year={2020}
}
```
