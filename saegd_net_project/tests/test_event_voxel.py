import numpy as np
from saegdnet.data.events import voxelize_events_np


def test_voxelize_empty():
    voxel, activity = voxelize_events_np(np.zeros((0, 4), np.float32), 16, 16, 3)
    assert voxel.shape == (6, 16, 16)
    assert activity.shape == (1, 16, 16)


def test_voxelize_some_events():
    events = np.array([[1, 2, 0, 1], [1, 2, 10, -1]], dtype=np.float32)
    voxel, activity = voxelize_events_np(events, 8, 8, 2)
    assert voxel.sum() > 0
    assert activity.sum() > 0
