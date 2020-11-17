PyTorch Partial Layer Freezing
=

The motivation for this repo is to allow PyTorch users to freeze only part of the layers in PyTorch.
It doesn't require any externat packages other than PyTorch itself.

Usage
-
1. Clone this repo.
2. Copy partial_freezing.py to folder, where you intend to run it.
3. Import partial_freezing into your .py file:
    ```python
    import partial_freezing
    ```
4. For a given layer (assume this is a Conv2d layer) do the following:
    ```python
    partial_freezing.freeze_conv2d_params(layer, indices)
    ```
    That will register a backward hook for a given parameters within the ```layer```, which will zero the gradients at specified ```indices```. In this case, ```indices``` is a list of integers, specifying which filters you intend to freeze.
  
Some more use cases can be found in test.py.
  
Limitations
-
Unfortunately, this code does not support freezing weights in ```Conv1d``` layers, because the backward pass is still inconsistent between ```Conv1d``` and other convolutional layers in PyTorch.
Please also note, our code freezes entire filters of convolutional layers, rather than specific weighs. We kept it this way to simplify the usage. If you want us to extend the functionality of our code, feel free to write to us, and we will be happy to do so.