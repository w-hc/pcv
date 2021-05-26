# Pixel Consensus Voting for Panoptic Segmentation (CVPR 2020)

The core of our approach, Pixel Consensus Voting, is a framework for instance segmentation based on the Generalized Hough transform. Pixels cast discretized, probabilistic votes for the likely regions that contain instance centroids. At the detected peaks that emerge in the voting heatmap, backprojection is applied to collect pixels and produce instance masks. Unlike a sliding window detector that densely enumerates object proposals, our method detects instances as a result of the consensus among pixel-wise votes. We implement vote aggregation and backprojection using native operators of a convolutional neural network. The discretization of centroid voting reduces the training of instance segmentation to pixel labeling, analogous and complementary to FCN-style semantic segmentation, leading to an efficient and unified architecture that jointly models things and stuff. We demonstrate the effectiveness of our pipeline on COCO and Cityscapes Panoptic Segmentation and obtain competitive results. 

## Quick Intro
- The codebase contains the essential ingredients of PCV, including various spatial discretization schemes and convolutional backprojection inference. The network backbone is a simple FPN on ResNet.
- Visualzier 1 (<code>src/vis.py</code>): loads a single image into a dynamic, interacive interface that allows users to click on pixels to inspect model prediction. It is built on matplotlib interactive API and jupyter widgets. Under the hood it's React.
- Visualizer 2 (<code>src/pan_vis.py</code>): A global inspector that take panoptic segmentation prediction and displays prediction segments against ground truth. Useful to track down which images make the most serious error and how.

- The core of PCV is contained in <code>src/pcv</code>. The results reported in the paper uses <code>src/pcv/pcv_basic</code>. There are also a few modification ideas that didn't work out e.g. "inner grid collapse" (<code>src/pcv/pcv_igc</code>), erasing boundary loss <code>src/pcv/pcv_boundless</code>, smoothened gt assignment <code>src/pcv/pcv_smooth</code>.
- The deconv voting filter weight intializaiton is in <code>src/pcv/components/ballot.py</code>. Different deconv discretization schemes can be found in <code>src/pcv/components/grid_specs.py</code>. <code>src/pcv/components/snake.py</code> manages the generation of snake grid on which pcv operates.

- The backprojection code is in <code>src/pcv/inference/mask_from_vote.py</code>. Since this is a non-standard procedure of convolving a filter to do equality comparison, I implemented a simple conv using advanced indexing. See the function <code>src/pcv/inference/mask_from_vote.py:unroll_img_inds</code>. For a fun side-project, I am thinking about rewriting the backprojection in Julia and generate GPU code (ptx) directly through LLVM. That way we don't have to deal with CUDA kernels that are hard to maintain. 
- The main entry point is <code>run.py</code> and <code>src/entry.py</code>


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
