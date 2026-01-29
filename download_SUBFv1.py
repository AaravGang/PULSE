import kagglehub

# Download latest version
path = kagglehub.dataset_download("sumairaziz/subf-v1-0-dataset-bearing-fault-vibration-data")

print("Path to dataset files:", path)