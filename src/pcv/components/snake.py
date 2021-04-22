"""
This module manages the generation of snake grid on which pcv operates.

It is intended to be maximally flexible by taking in simple parameters to
achieve various grid configurations for downstream experimentation.
"""
import numpy as np


class Snake():
    def __init__(self):
        pass

    @staticmethod
    def flesh_out_grid_spec(raw_spec):
        """
        Args:
            raw_spec: [N, 2] each row (size, num_rounds)
        """
        if not isinstance(raw_spec, np.ndarray):
            raw_spec = np.array(raw_spec)
        shape = raw_spec.shape
        assert len(shape) == 2 and shape[0] > 0 and shape[1] == 2
        size = raw_spec[0][0]
        trail = [ np.array([0, 0, (size - 1) // 2]).reshape(1, -1), ]
        field_diam = size
        for size, num_rounds in raw_spec:
            for _ in range(num_rounds):
                field_diam, _round_trail = Snake.ring_walk(field_diam, size)
                trail.append(_round_trail)
        trail = np.concatenate(trail, axis=0)
        return field_diam, trail

    @staticmethod
    def ring_walk(field_diam, body_diam):
        assert body_diam > 0 and body_diam % 2 == 1
        body_radius = (body_diam - 1) // 2

        assert field_diam > 0 and field_diam % 2 == 1
        field_radius = (field_diam - 1) // 2

        assert field_diam % body_diam == 0

        ext_diam = field_diam + 2 * body_diam
        ext_radius = field_radius + body_diam
        assert ext_diam == ext_radius * 2 + 1

        j = 1 + field_radius + body_radius
        # each of the corner coord is the offset from field center
        # anticlockwise SE -> NE -> NW -> SW ->
        corner_centers = np.array([(+j, +j), (-j, +j), (-j, -j), (+j, -j)])
        directs = np.array([(-1, 0), (0, -1), (+1, 0), (0, +1)])
        trail = []
        num_tiles = ext_diam // body_diam
        for corn, dirc in zip(corner_centers, directs):
            segment = [corn + i * dirc * body_diam for i in range(num_tiles - 1)]
            trail.extend(segment)
        trail = np.array(trail)  # [N, 2] each row is a center-based offset
        sizes = np.array([body_radius] * len(trail)).reshape(-1, 1)
        trail = np.concatenate([trail, sizes], axis=1)
        return ext_diam, trail  # [N, 3]: (offset_y, offset_x, radius)

    @staticmethod
    def paint_trail_mask(field_diam, trail, tiling=True):
        """
        Args:
            trail: [N, 3] each row (offset_y, offset_x, radius)
        """
        assert field_diam > 0 and field_diam % 2 == 1
        field_radius = (field_diam - 1) // 2
        CEN = np.array((field_radius, field_radius))
        trail = trail.copy()
        trail[:, :2] += CEN
        canvas = -1 * np.ones((field_diam, field_diam), dtype=int)
        for i, walk in enumerate(trail):
            y, x, r = walk
            if tiling:
                y, x = y - r, x - r
                d = 2 * r + 1
                canvas[y: y + d, x: x + d] = i
            else:
                canvas[y, x] = i
        return canvas

    @staticmethod
    def paint_bound_ignore_trail_mask(field_diam, trail):
        """
        Args:
            trail: [N, 3] each row (offset_y, offset_x, radius)
        """
        assert field_diam > 0 and field_diam % 2 == 1
        field_radius = (field_diam - 1) // 2
        CEN = np.array((field_radius, field_radius))
        trail = trail.copy()
        trail[:, :2] += CEN
        canvas = -1 * np.ones((field_diam, field_diam), dtype=int)
        for i, walk in enumerate(trail):
            y, x, r = walk
            d = 2 * r + 1
            boundary_ignore = int(d * 0.12)
            # if d > 1:  # at least cut 1 unless the grid is of size 1
            #     boundary_ignore = max(boundary_ignore, 1)
            y, x = y - r + boundary_ignore, x - r + boundary_ignore
            d = d - 2 * boundary_ignore
            canvas[y: y + d, x: x + d] = i
        return canvas

    @staticmethod
    def vote_channel_splits(raw_spec):
        splits = []
        acc_size = raw_spec[0][0]  # inner most size
        for i, (size, num_rounds) in enumerate(raw_spec):
            inner_blocks = acc_size // size
            total_blocks = inner_blocks + 2 * num_rounds
            acc_size += 2 * num_rounds * size
            prepend = (i == 0)
            num_chnls = (total_blocks ** 2 - inner_blocks ** 2) + int(prepend)
            splits.append(num_chnls)
        return splits


if __name__ == '__main__':
    sample_spec = np.array([
        # (size, num_rounds)
        (1, 1),
        (3, 1),
        (9, 1)
    ])
    s = Snake()
    diam, trail = s.flesh_out_grid_spec(sample_spec)
    mask = s.paint_trail_mask(diam, trail, tiling=True)
    print(mask.shape)
