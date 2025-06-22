# Aerial Sheep Detection with YOLOv12

This project uses YOLOv12 for sheep detection in aerial images, featuring a complete K-fold cross-validation and model evaluation pipeline.

⚠️ **Important Notice**: This project uses a third-party improved YOLOv12 implementation ([sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12)), not the official Ultralytics version. The official Ultralytics currently has YOLO11 as their latest version, with no official YOLOv12 release. This third-party version claims to fix training stability and memory efficiency issues.

## 📁 Project Structure

```
yolov12/
├── aerial_sheep/          # Sheep detection dataset
│   ├── train/
│   │   ├── images/        # Training images
│   │   └── labels/        # Training labels
│   ├── valid/
│   │   ├── images/        # Validation images
│   │   └── labels/        # Validation labels
│   └── test/
│       ├── images/        # Testing images
│       └── labels/        # Testing labels
├── cv.py                  # Main training script
├── cv.ipynb              # Jupyter Notebook version
├── ultralytics/          # YOLOv12 core code
├── app.py               # Application entry point
├── requirements.txt      # Dependency list
└── README.md            # Project documentation
```

## 🔧 Environment Setup

### 1. Download Third-party YOLOv12
⚠️ **Note**: This is not the official Ultralytics version
```bash
git clone https://github.com/sunsmarterjie/yolov12.git yolov12
cd yolov12
```

### 2. Install Dependencies
Configure Python environment according to the original YOLOv12 project README:

```bash
# Download flash-attention (if needed)
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Create virtual environment
conda create -n yolov12 python=3.11
conda activate yolov12

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 3. Dataset Preparation
Ensure the `aerial_sheep` folder is located in the project root directory with complete training, validation, and testing data.

## 🚀 Usage

### Method 1: Python Script
```bash
python cv.py
```

### Method 2: Jupyter Notebook
```bash
jupyter notebook cv.ipynb
```

## 📊 Experiment Pipeline

The program automatically executes the following 4 stages:

### Stage 1: Data Merging
- Merges `train` and `valid` data into the `new_train` directory
- Prepares data for K-fold cross-validation

### Stage 2: K-Fold Cross-Validation
- Uses 5-fold cross-validation to assess model stability
- Outputs F1, Precision, Recall, and mAP metrics for each fold
- Calculates average performance metrics

### Stage 3: Final Model Training
- Trains the final model using all merged data
- Saves best weights to `final_sheep_model/best/weights/best.pt`

### Stage 4: Test Set Evaluation
- Evaluates final model performance on independent test set
- Outputs final detection metrics

## 📈 Expected Results

### K-Fold Cross-Validation Results Example
```
========================================
K-Fold Cross-Validation Summary
========================================
Average F1 score     : 0.8450
Average Precision    : 0.8200
Average Recall       : 0.8700
Average mAP@0.5      : 0.9100
Average mAP@0.5:0.95 : 0.7200
```

### Final Test Set Evaluation
```
========================================
Final Test Set Evaluation
========================================
Test F1 score     : 0.8500
Test Precision    : 0.8300
Test Recall       : 0.8600
Test mAP@0.5      : 0.9200
Test mAP@0.5:0.95 : 0.7300
```

## 🎯 Model Configuration

- **Model Architecture**: YOLOv12-turbo (Third-party improved version)
- **Project Source**: [sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12)
- **Improvements**: Fixes training stability, memory efficiency, and performance issues
- **Input Size**: 640x640
- **Training Epochs**: 75 epochs for K-fold validation, 100 epochs for final training
- **Batch Size**: 6
- **Number of Classes**: 1 (sheep)

## 📝 Notes

1. Ensure CUDA environment is properly configured (if using GPU training)
2. The program requires significant runtime; recommended to run on servers or high-performance machines
3. Final model weights are saved in the `final_sheep_model` directory
4. To adjust hyperparameters, modify the corresponding parameters in `cv.py`
5. **Version Notice**: This project uses a third-party YOLOv12 implementation, different from official Ultralytics versions

## 🤝 Usage Instructions

This project is based on the third-party YOLOv12 framework, specifically optimized for sheep detection tasks in aerial images. Through a complete cross-validation pipeline, it ensures model stability and generalization capability.

Upon completion, you will obtain:
- Detailed K-fold cross-validation report
- Trained final detection model
- Performance evaluation results on the test set

## 🔗 Related Links

- [Third-party YOLOv12 Project](https://github.com/sunsmarterjie/yolov12)
- [Official Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- [YOLO Model Comparison](https://www.ultralytics.com/blog/comparing-ultralytics-yolo11-vs-previous-yolo-models)