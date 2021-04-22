import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import Output
from matplotlib.patches import Circle

from panoptic.pcv.gaussian_smooth.prob_tsr import MakeProbTsr
from panoptic.pcv.components.snake import Snake

from panoptic.vis import Visualizer as BaseVisualizer


class Plot():
    def __init__(self, ax, bin_center_yx, vote_mask, spatial_prob):
        self.ax = ax
        self.bin_center_yx = bin_center_yx
        self.vote_mask = vote_mask
        self.spatial_prob = spatial_prob

        # self.pressed_xy = None
        self.dot = None
        self.texts = None
        self.init_artists()
        self.render_visual()

    def init_artists(self):
        if self.dot is not None:
            self.dot.remove()

        if self.texts is not None:
            assert isinstance(self.texts, (tuple, list))
            for elem in self.texts:
                elem.remove()

        self.dot = None
        self.texts = None

    def render_visual(self):
        self.ax.imshow(self.vote_mask)

    def press_coord(self, x, y, button):
        del button  # ignoring button for now
        # self.pressed_xy = x, y
        self.init_artists()
        self.render_single_dot(x, y)
        self.render_prob_dist(x, y)

    def render_prob_dist(self, x, y):
        thresh = 0
        dist = self.spatial_prob[y, x]
        inds = np.where(dist > thresh)[0]
        probs = dist[inds] * 100
        # print(probs)
        bin_centers = self.bin_center_yx[inds]

        acc = []
        for cen, p in zip(bin_centers, probs):
            y, x = cen
            _a = self.ax.text(
                x, y, s='{:.2f}'.format(p), fontsize='small', color='r'
            )
            acc.append(_a)
        self.texts = acc

    def query_coord(self, x, y, button):
        pass

    def motion_coord(self, x, y):
        self.press_coord(x, y, None)

    def render_single_dot(self, x, y):
        cir = Circle((x, y), radius=0.5, color='white')
        self.ax.add_patch(cir)
        self.dot = cir


class Visualizer(BaseVisualizer):
    def __init__(self):
        spec = [  # 243, 233 bins
            (3, 4), (9, 3), (27, 3)
        ]
        # spec = [
        #     (3, 3), (7, 3), (21, 4)
        # ]
        diam, grid_spec = Snake.flesh_out_grid_spec(spec)
        vote_mask = Snake.paint_trail_mask(diam, grid_spec)
        maker = MakeProbTsr(spec, diam, grid_spec, vote_mask)
        spatial_prob = maker.compute_voting_prob_tsr()

        self.vote_mask = vote_mask
        self.spatial_prob = spatial_prob

        radius = (diam - 1) // 2
        center = np.array((radius, radius))
        self.bin_center_yx = grid_spec[:, :2] + center

        self.output_widget = Output()
        self.init_state()
        self.pressed = False
        np.set_printoptions(
            formatter={'float': lambda x: "{:.3f}".format(x)}
        )

    def vis(self):
        fig = plt.figure(figsize=(10, 10), constrained_layout=True)
        self.fig = fig
        self.canvas = fig.canvas
        self.plots = dict()

        key = 'spatial prob dist'
        ax = fig.add_subplot(111)
        ax.set_title(key)
        self.plots[key] = Plot(
            ax, self.bin_center_yx, self.vote_mask, self.spatial_prob
        )
        self.connect()


def test():
    # spec = [  # 243, 233 bins
    #         (1, 1), (3, 4), (9, 3), (27, 3)
    # ]
    spec = [  # 243, 233 bins
        (3, 1), (9, 1)
    ]
    diam, grid_spec = Snake.flesh_out_grid_spec(spec)
    vote_mask = Snake.paint_trail_mask(diam, grid_spec)
    maker = MakeProbTsr(spec, diam, grid_spec, vote_mask)


if __name__ == "__main__":
    test()
