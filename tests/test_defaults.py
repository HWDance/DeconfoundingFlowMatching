from deconfoundingfm import DeconfoundingFM, DeconfoundingFMConfig


def test_vector_public_defaults():
    cfg = DeconfoundingFMConfig()
    assert cfg.hidden == 64
    assert cfg.layers == 1
    assert cfg.iterations == 10_000
    assert cfg.nuisance_hidden == 64
    assert cfg.nuisance_layers == 1

    model = DeconfoundingFM(cfg)
    opts = model._resolved_training_options("mlp")
    assert opts["lr"] == 1e-4
    assert opts["batch_size"] == 256
    assert opts["plugin_reservoir"] == 64
    assert opts["plugin_batch"] == 4


def test_image_specific_defaults_are_preserved():
    model = DeconfoundingFM(DeconfoundingFMConfig())
    opts = model._resolved_training_options("unet")
    assert opts["lr"] == 1e-4
    assert opts["batch_size"] == 64
    assert opts["plugin_reservoir"] == 1
    assert opts["plugin_batch"] == 1
