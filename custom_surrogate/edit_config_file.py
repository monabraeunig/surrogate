import json

## Read the JSON file
with open("custom_surrogate.json", "r") as f:
    data = json.load(f)

## Shape: torch.Size([output size, 1, input size])
data["lengthscale"] =[[[0.1151, 0.2556]]]
with open(json_file_path, "w") as f:
    json.dump(data, f, indent=2)

## Shape: torch.Size([output size])
data["outputscale"] = [9.9763]
with open(json_file_path, "w") as f:
    json.dump(data, f, indent=2)

## Shape: torch.Size([output size])
data["mean"] = [-6.7727]
with open(json_file_path, "w") as f:
    json.dump(data, f, indent=2)

## Shape: [1]
data["input_dim"] = 2
with open(json_file_path, "w") as f:
    json.dump(data, f, indent=2)

## Shape: [1]
data["output_dim"] = 1
with open(json_file_path, "w") as f:
    json.dump(data, f, indent=2)
