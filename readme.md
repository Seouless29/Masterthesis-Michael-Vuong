# Implementation and Visualization of Learnable Parameters in Gabor Convolutional Networks

This project consists of various models used during the master thesis, a training loop with its pre run file to check if the used models are set up correctly. Furthermore several statistics generating files have been included in order to visualize the results.


## Structure

The files have been rearranged such that it is possible to plug in your own model. 
In the training folder, there are two files. The prerun file is there to check if the model in questions has the required parameters, while the train file is the main training loop.

In the folder statistics there are several files.
Statistics creates four plots regarding performances between the three models.
test_run is the file to run the model performance on the test set.
wavelet_evolution shows the wavelet and their respective parameters over time. In this implementation I only look at the first and the best epoch. This can be easily extended to incorporate more timesteps.
At last artifacts consists of the recreation of the original image from the output of the gabor layer. Additionally I also show what the output of the gabor layer is by creating feature maps. Finally as a bonus I also show in each layer what the model currently is evaluating.

## Remarks

The outputs shown here are still run with the old models. I have updated the code in general such that it is cleaner and  are easier to use, therefore the results may vary with a rerun. The model that have been used for the results are gabor_cnn_mode_dropout.py (non-dilated) and gabor_big_model.py (dilated).
model_uniform.py now has both version of the model where I can set it to dilation mode or non dilation mode. The configuration can also be changed in the same file.

The dataset is from https://data.mendeley.com/datasets/x4dwwfwtw3/3. 
The original has images in tiff format which were problematic during preprocessing. That is why I converted them into PyTorch tensor files.