import os
import cv2
import numpy as np
import pandas as pd
import math

# Paths
catalog_file = "LU1319373_Wang & Wu_2021.txt"
tif_dir = "R-Value Maps"
dataset_dir = "yolo_dataset"
images_train = os.path.join(dataset_dir, "images", "train")
images_val = os.path.join(dataset_dir, "images", "val")
labels_train = os.path.join(dataset_dir, "labels", "train")
labels_val = os.path.join(dataset_dir, "labels", "val")

for p in [images_train, images_val, labels_train, labels_val]:
    os.makedirs(p, exist_ok=True)

print("Loading catalog...")
columns = ["ID", "Longitude", "Latitude", "Diameter_m", "Depth_m", "Source", "Source_lon", "Source_lat", "Source_dia"]
# Skip the first 16 lines of the dataset which are comments/headers
df = pd.read_csv(catalog_file, sep='\t', skiprows=16, names=columns, low_memory=False)
# Convert relevant columns to numeric, skipping any malformed rows
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
df['Diameter_m'] = pd.to_numeric(df['Diameter_m'], errors='coerce')
df = df.dropna(subset=['Longitude', 'Latitude', 'Diameter_m'])

print(f"Loaded {len(df)} craters.")

# Function to match TIF files to diameter bins if they are specified in the filename
def extract_diameter_range(filename):
    # Example: R-value_D_10_14.1.tif -> 10, 14.1
    name = filename.replace("R-value_D_", "").replace(".tif", "")
    parts = name.split("_")
    try:
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
    except:
        pass
    return None, None

tif_files = [f for f in os.listdir(tif_dir) if f.endswith(".tif")]

print(f"Processing {len(tif_files)} TIF files...")
train_split = 0.8

for idx, tif_file in enumerate(tif_files):
    filepath = os.path.join(tif_dir, tif_file)
    print(f"Processing {tif_file}...")
    
    # Load TIF using cv2
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to read {tif_file}")
        continue
    
    # Normalize to 0-255 uint8
    img_valid = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    norm_img = cv2.normalize(img_valid, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    h, w = norm_img.shape
    
    # Check if this image corresponds to a specific diameter bin
    min_d, max_d = extract_diameter_range(tif_file)
    
    # Filter craters
    if min_d is not None and max_d is not None:
        # Convert min_d, max_d to meters
        mask = (df['Diameter_m'] >= min_d * 1000) & (df['Diameter_m'] <= max_d * 1000)
        craters = df[mask]
    else:
        craters = df
        
    # Split
    is_train = idx < len(tif_files) * train_split
    img_dir = images_train if is_train else images_val
    lbl_dir = labels_train if is_train else labels_val
    
    base_name = tif_file.replace(".tif", ".png")
    lbl_name = tif_file.replace(".tif", ".txt")
    
    # Save Image
    cv2.imwrite(os.path.join(img_dir, base_name), norm_img)
    
    # Generate Labels
    with open(os.path.join(lbl_dir, lbl_name), "w") as f:
        for _, row in craters.iterrows():
            lon = row['Longitude']
            lat = row['Latitude']
            dia_m = row['Diameter_m']
            
            # Normalize coordinates (0 to 1)
            # Longitude: -180 to 180 -> 0 to 1
            x_norm = (lon + 180.0) / 360.0
            
            # Latitude: 90 to -90 -> 0 to 1 (top to bottom)
            # 90 is top (0), -90 is bottom (1)
            y_norm = (90.0 - lat) / 180.0
            
            # Diameter to normalized width/height
            # 1 degree of latitude on Moon ~ 30.32 km (1737.4 km radius)
            dia_km = dia_m / 1000.0
            h_norm = (dia_km / 30.32) / 180.0
            
            # Longitude degree size shrinks with latitude
            lat_rad = math.radians(lat)
            w_norm = (dia_km / (30.32 * max(0.1, math.cos(lat_rad)))) / 360.0
            
            # Ensure valid bounding box
            x_norm = max(0.001, min(0.999, x_norm))
            y_norm = max(0.001, min(0.999, y_norm))
            
            # Ensure we don't have 0 width or height (which YOLO hates)
            w_norm = max(0.001, w_norm)
            h_norm = max(0.001, h_norm)
            
            # YOLO format: class x_center y_center width height
            # Craters will be class 0
            f.write(f"0 {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

# Create dataset.yaml
yaml_content = f"""path: {os.path.abspath(dataset_dir)}
train: images/train
val: images/val

names:
  0: crater
"""
with open("dataset.yaml", "w") as f:
    f.write(yaml_content)

print("Dataset prepared. To train YOLOv8, run:")
print("yolo task=detect mode=train data=dataset.yaml model=yolov8n.pt epochs=10 imgsz=1280")
