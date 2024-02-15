import json

json_file_path = "custom_surrogate.json"

# Read the JSON file
with open(json_file_path, "r") as f:
    data = json.load(f)

data["model_port"] = 'http://0.0.0.0:4243'

data["model_name"] = "posterior"

## level of certainty gp must hold to accept prediction
data["threshold"] = 0.0001

## Shape: [output size, 1, input size]
## except if :
## input size == 1 and ouput size == 1 => [1,1]
## input size == 2 and ouput size == 1 => [1,2]
data["lengthscale"] =[[0, 0]]

## Shape: [output size]
data["outputscale"] = [0]

## Shape: [output size]
data["mean"] = [0]

## set to True if variance plot is desired
data["plot"] = False
## set lower and upper bounds for the x any y axis in the plot
data["lower_bound_x"] = -6
data["lower_bound_y"] = -6
data["upper_bound_x"] = 6
data["upper_bound_y"] = 6

with open(json_file_path, "w") as f:
    json.dump(data, f, indent=2)
