def test_pytorch_install():
    import torch
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() > 0