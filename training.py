from ultralytics import YOLO
import multiprocessing

def main():
    model = YOLO(r'Desktop/yolov12/ultralytics/cfg/models/v12/yolov12.yaml')
    # Train the model
    results = model.train(
      data='data.yaml',
      epochs=100, 
      batch=2,
      imgsz=640,
      scale=0.5,
      mosaic=0.0,
      mixup=0.0,
      copy_paste=0.1,
      device="0",
      save=True,
      project="sheep_detection"
    )

    # 评估模型性能并获取F1分数
    metrics = model.val(verbose=True,save_json=True)  # 使用verbose=True参数查看详细的评估指标

    # 打印F1分数
    print(f"F1 score: {metrics.box.f1}")  # 直接访问指标中的f1属性

    
    # Perform object detection on an image
    results = model(r"Desktop/yolov12/aerial_sheep/train/images/DJI_0004_0254_jpg.rf.454671f008ff5771b657fc7b11d37f15.jpg")
    results[0].show()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()