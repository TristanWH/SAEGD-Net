def test_imports():
    import saegdnet
    from saegdnet.models.saegd import build_model
    from saegdnet.diffusion.gaussian import GaussianDiffusion
    assert saegdnet.__version__
