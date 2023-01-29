import onets_lab


class dummy_class(object):
    def __init__(self, **kwargs):
        pass

    def compute_tendency(self):
        pass


def test_init():
    bm_sim = onets_lab.box_model_simulation(
        dummy_class,
    )
    assert isinstance(bm_sim, onets_lab.box_model_simulation)


def test_single_box_init():
    obj = onets_lab.single_box()
    assert isinstance(obj, onets_lab.single_box)
