import torch
from mlp import MNISTMLP

def test_forward_shape():
    model = MNISTMLP()
    x = torch.randn(4, 1, 28, 28)
    logits = model(x)
    assert logits.shape == (4, 10)

def test_train_step():
    model = MNISTMLP()
    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    loss = torch.nn.CrossEntropyLoss()(model(x), y)
    loss.backward()  # should not raise
