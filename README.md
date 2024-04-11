# Surrogate
`surrogate.py` is a surrogate microservice applicable to any UM-Bridge model. It connects to the model and works as follows:

A UM-Bridge connects to the surrogate on port `4244` and sends an input request. Subsequently, the surrogate hands the input to a Gaussian process that computes the posterior distribution consisting of a mean and a variance. The mean is a prediction for the output, while the variance provides the level of uncertainty this prediction holds.

If the variance falls below a predefined threshold, the surrogate categorizes the prediction as reliable and returns the mean. If the variance surpasses the predefined threshold, the surrogate rejects the prediction and declares it as unreliable. Consequently, the surrogate passes the input onto the underlying UM-Bridge model for an actual computation of the output. Once the model completes its computation, it returns the output to the surrogate. The surrogate then returns this output and uses the newly made observation to further train the Gaussian process.

The Gaussian process comes from the library BoTorch.

The surrogate regularly writes checkpoints into the file `checkpoint.pth`. These contain all observations computed by the UM-Bridge model. The program can be terminated at any state and restarted through the checkpoint. 

To incorporate existing observations, put them in a file called `data.txt` and place it next to `surrogate.py`. 

## Required installations

To run the surrogate, the following libraries must be installed:
* BoTorch
* GPyTorch 
* Matplotlib
* NumPy
* PyTorch
* UM-Bridge

## Configurations
Before using the surrogate, the following model-specific configurations must be specified in custom_surrogat.json:
* model_name: The name of the UM-Bridge model
* model_port: The port on which the model runs
* threshold: A threshold value for the variance. Set to the highest level of certainty to accept a prediction
* custom_hyperparameters: If set to False, the surrogate performs hyperparameter optimization. If set to False, the hyperparameters must be set manually.
* plot: If set to True, the surrogate creates a visualization of the variance as a heatmap after each training session. This feature is only supported for models with a two-dimensional input and a one-dimensional output

Optional configurations:

If custom_hyperparameters is True, the following three hyperparameter values must be set:
* mean: A value for a constant mean. (Shape: [output size])
* outputscale: A value to sacle the Matern Kernel with. (Shape: [output size])
* lengthscale: A value for the length scale in a Matern Kernel. (Shape: [output size, 1, input size] except if: input size == 1 and output size == 1 => Shape: [1,1] or if: input size == 2 and output size == 1 => Shape: [1,2])

If plot is True, the following values must be set:
* lower_bound_x: A lower bound for the x-axis of the plot.
* lower_bound_y: A lower bound for the y-axis of the plot.
* upper_bound_x: An upper bound for the x-axis of the plot.
* upper_bound_y: An upper bound for the y-axis of the plot.

Example:
```
{
  "model_name": "posterior",
  "model_port": "http://0.0.0.0:4243",
  "threshold": 0.001,
  "custom_hyperparameters": true,
  "plot": true

  "lengthscale": [[0.4593,0.5052]],
  "outputscale": [16.3747],
  "mean": [-7.7317],

  "lower_bound_x": -6,
  "lower_bound_y": -6,
  "upper_bound_x": 6,
  "upper_bound_y": 6
}
```
