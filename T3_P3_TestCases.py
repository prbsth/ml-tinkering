import numpy as np
import torch
def test_forward_pass(net):
    sample_image = torch.zeros(1, 1, 28, 28)
    output = net(sample_image)
    assert output.shape == (1, 10), "Output shape is incorrect. Expected (1, 10), got {}".format(output.shape)

    print("Forward pass test passed!")
