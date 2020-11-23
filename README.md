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
5. If you plan to re-use the function (change the indices of frozen layers), make sure, you save the handles to the backward hooks returned by ```freeze_conv2d_params()```, and pass them as arguments, when re-using the function in your code at the same layer. It can be done as follows:
    ```python
    weight_hook_handle, bias_hook_handle = partial_freezing.freeze_conv2d_params(layer, indices)
    (...) # your code
    new_weight_hook_handle, new_bias_hook_handle = partial_freezing.freeze_conv2d_params(layer, indices, weight_hook_handle=weight_hook_handle, bias_hook_handle=bias_hook_handle)
    ```
    This will ensure that your current hooks will be removed and the new will be added properly.
    
Some more use cases can be found in test.py.
  
Limitations
-
* Our code freezes entire filters of convolutional layers, rather than specific weighs. We kept it this way to simplify the usage. If you want us to extend the functionality of our code, feel free to write to us, and we will be happy to do so.
* Since the mechanism for updating weights in case of using weight decay is a bit different, the weights may still be changing if ```weight_decay > 0``` in your optimzier settings.