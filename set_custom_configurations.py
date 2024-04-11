import json

json_file_path = "custom_surrogate.json"

# Read the JSON file
with open(json_file_path, "r") as f:
    data = json.load(f)

data["model_port"] = 'http://0.0.0.0:4243'

data["model_name"] = "posterior"

# Highest level of certainty to accept a prediction
data["threshold"] = 0.0001

# True: Hyperparameters must be set manually below
# False: Surrogate does hyperparameter optimization
data["custom_hyperparameters"] = False


## Hyperparameter
# Shape: [output size, 1, input size]
# except if :
# input size == 1 and output size == 1 => [1,1]
# input size == 2 and output size == 1 => [1,2]
data["lengthscale"] =[[0, 0]]

# Shape: [output size]
data["outputscale"] = [0]

# Shape: [output size]
data["mean"] = [0]


# Set to True if variance plot is desired
# Only works for models with input size 2 and output size 1
data["plot"] = False

# Set lower and upper bounds for the x (first input) and y (second input) axis in the plot
data["lower_bound_x"] = -6
data["lower_bound_y"] = -6
data["upper_bound_x"] = 6
data["upper_bound_y"] = 6

with open(json_file_path, "w") as f:
    json.dump(data, f, indent=2)
