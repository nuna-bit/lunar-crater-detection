# Lunar Crater Detection using YOLOv8 🌑

With humanity's recent return to lunar orbit via the Artemis II mission, my eyes have been glued to the Moon. It got me thinking: with all the incredible orbital data we've collected over the years, why not run some insights of our own? 

This repository is my personal passion project in which I built a model to automatically detect and map lunar craters. By leveraging the state-of-the-art **YOLOv8** computer vision framework, I set out to teach a model how to spot geological formations across the lunar surface just like a planetary scientist would.

## Technologies I Used
- **Python** (Pandas, NumPy, OpenCV)
- **Computer Vision** (Ultralytics YOLOv8)
- **Geospatial Processing** (Handling raw orbital TIF image arrays)

## The Data Driving It
1. **R-Value Maps**: High-resolution, unstructured TIF arrays (capturing rock abundance metrics sourced from geological scans like the LRO Diviner).
2. **Crater Catalog**: Ground-truth labels from the *Wang & Wu (2021) Lunar Global Crater Catalog*. I used this to bridge physical geographic locations (Longitude/Latitude/Diameter) directly into bounding boxes my AI could understand.

## How I Built the Architecture

1. **Dataset Synthesis (`prepare_dataset.py`)**: 
   Since object detection models need normalized images rather than raw geographical coordinates, I wrote a script to handle the heavy translation. 
   - It ingests massive floating-point `.tif` images via OpenCV, normalizing them into standard 0-255 `uint8` visual PNG images.
   - It automatically filters my catalog database based on physical crater dimensions.
   - It mathematically translates spherical moon mapping limits (Degrees & Kilometers) into YOLO coordinate structures (normalized `x_center`, `y_center`, `width`, `height`).

2. **YOLO Deep Learning Engine**: 
   Once the dataset is engineered, YOLO handles the neural network construction, extracting features iteratively to locate craters and output precision visual inferences!

## Want to Run It Yourself?

**1. Install the dependencies**
```bash
pip install ultralytics opencv-python pandas numpy
```

**2. Prepare the annotated dataset**
Run my synthesis script to link the GeoTiffs and text catalogs into the `yolo_dataset` directory:
```bash
python prepare_dataset.py
```

**3. Train the model**
Initiate the deep learning training using the CLI loop:
```bash
yolo task=detect mode=train data=dataset.yaml model=yolov8n.pt epochs=50 imgsz=640
```

**4. Run Predictions**
Perform inference to locate craters on new sections of the orbital maps:
```bash
yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source="R-Value Maps/R-value_D_2.5_3.5.tif" save=True
```