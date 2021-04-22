## Implementation for Pixel Consensus Voting (CVPR 2020).
---
This codebase contains
- the essential ingredients of PCV, including various spatial discretization schemes and convolutional backprojection inference. The network backbone is a simple FPN on ResNet.
- Visualzier 1: loads a single image into a dynamic, interacive interface that allows users to click on pixels to inspect model prediction. It is built on matplotlib interactive API and jupyter widgets. Under the hood it's React.
- Visualizer 2: A global inspector that take panoptic segmentation prediction and displays prediction segments against ground truth. Useful to track down which images make the most serious error and how.


### Codebase walkthrough

- The core of PCV is contained in <code>src/pcv</code>. The results reported in the paper uses <code>src/pcv/pcv_basic</code>. There are also a few modification ideas that didn't work out e.g. "inner grid collapse" (<code>src/pcv/pcv_igc</code>), erasing boundary loss <code>src/pcv/pcv_boundless</code>, smoothened gt assignment <code>src/pcv/pcv_smooth</code>.
- The deconv voting filter weight intializaiton is in <code>src/pcv/components/ballot.py</code>. Different deconv discretization schemes can be found in <code>/src/pcv/components/grid_specs.py</code>. <code>src/pcv/components/snake.py</code> manages the generation of snake grid on which pcv operates.

- The backprojection code is in <code>src/pcv/inference/mask_from_vote.py</code>. Since this is a non-standard procedure of convolving a filter to do equality comparison, I implemented a simple conv using advanced indexing. See the function <code>src/pcv/inference/mask_from_vote.py:unroll_img_inds</code>.
- The main entry point is <code>/run.py</code> and <code>src/entry.py</code>
- The rest of the codebase are pretty self explanatory. Please email me if you have any questions.
