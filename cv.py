import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import multiprocessing
from sklearn.model_selection import KFold
import tempfile
os.environ['TQDM_DISABLE'] = '1'  # 在脚本开头添加这行

cur_dir = Path('aerial_sheep')  # 修改：指向aerial_sheep目录

def merge_files(cur_dir):
    cur_dir = Path(cur_dir)
    train_img = cur_dir/'train'/'images'
    train_label = cur_dir/'train'/'labels'
    val_img = cur_dir/'valid'/'images'
    val_label = cur_dir/'valid'/'labels'

    new_train_img = list(train_img.glob('*.jpg')) + list(val_img.glob('*.jpg'))
    new_train_label = list(train_label.glob('*.txt'))+ list(val_label.glob('*.txt'))
    new_train_img = sorted(new_train_img)
    new_train_label = sorted(new_train_label)

    new_img_path = cur_dir/'new_train'/'images'
    new_label_path = cur_dir/'new_train'/'labels'
    new_img_path.mkdir(parents=True, exist_ok=True)
    new_label_path.mkdir(parents=True, exist_ok=True)
    for i in range(len(new_train_img)):
        img = new_train_img[i]
        label = new_train_label[i]
        if img.stem != label.stem:
            print(f"警告：文件名不匹配 - 图片:{img.name}, 标签:{label.name}")
        shutil.copy(img,new_img_path)
        shutil.copy(label,new_label_path)

    return 1

def k_fold_train(data_dir, k, seed):
    data_dir = Path(data_dir)
    img_path = data_dir/'new_train'/'images'  # ✅ 使用merge后的路径
    label_path = data_dir/'new_train'/'labels'
    imgs = sorted(os.listdir(img_path))
    labels = sorted(os.listdir(label_path))
    pairs = list(zip(imgs, labels))

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    all_metrics = []
    for fold, (train_index, test_index) in enumerate(kf.split(pairs)):
        print(f"\nFold {fold+1} 开始...")  # 添加进度显示
        train_pairs = [pairs[i] for i in train_index]
        test_pairs = [pairs[i] for i in test_index]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            train_img_dir = temp_path/'train'/'images'
            train_label_dir = temp_path/'train'/'labels'
            test_img_dir = temp_path/'valid'/'images'  # 修正：改为valid
            test_label_dir = temp_path/'valid'/'labels'

            train_img_dir.mkdir(parents=True)  # 修正：添加parents=True
            train_label_dir.mkdir(parents=True)
            test_img_dir.mkdir(parents=True)
            test_label_dir.mkdir(parents=True)
            
            for (p_img, p_label) in train_pairs:
                shutil.copy(img_path / p_img, train_img_dir)  # 修正：使用完整路径
                shutil.copy(label_path / p_label, train_label_dir)

            for (pt_img, pt_label) in test_pairs:
                shutil.copy(img_path / pt_img, test_img_dir)  # 修正：使用完整路径
                shutil.copy(label_path / pt_label, test_label_dir)

            # 修正：直接创建yaml文件
            yaml_path = temp_path / 'data.yaml'
            with open(yaml_path, 'w') as f:
                f.write(f"""
train: {train_img_dir.parent}
val: {test_img_dir.parent}
nc: 1
names: ['sheep']
""".strip())

            model = YOLO(r'/home/z5470186/Desktop/yolov12/ultralytics/cfg/models/v12/yolov12s.yaml')
            results = model.train(
                data=str(yaml_path),  # 修正：使用字符串路径
                epochs=75,
                batch=6,
                imgsz=640,
                scale=0.5,
                mosaic=1,
                mixup=0.0,
                copy_paste=0,
                device="0",
                save=False,  # 修正：不保存模型
                project="cv_sheep_detection",
                name=f"fold_{fold+1}",
                verbose=False,
                workers=2
            )

            metrics = model.val(data=str(yaml_path), verbose=True, save_json=True)
            all_metrics.append(metrics.box)
            
            # 简单直接：如果是数组就取第一个元素
            print(f"Fold {fold+1} F1: {metrics.box.f1[0]:.4f}")

    # 修正：移到循环外面
    print(f"\n{'=' * 40}")
    print("K-Fold 交叉验证结果汇总")
    print(f"{'=' * 40}")

    # 简单直接：计算平均值时都用[0]
    avg_f1 = sum(m.f1[0] for m in all_metrics) / k
    avg_precision = sum(m.mp for m in all_metrics) / k
    avg_recall = sum(m.mr for m in all_metrics) / k
    avg_map50 = sum(m.map50 for m in all_metrics) / k
    avg_map = sum(m.map for m in all_metrics) / k

    print(f"平均 F1 score     : {avg_f1:.4f}")
    print(f"平均 Precision    : {avg_precision:.4f}")
    print(f"平均 Recall       : {avg_recall:.4f}")
    print(f"平均 mAP@0.5      : {avg_map50:.4f}")
    print(f"平均 mAP@0.5:0.95 : {avg_map:.4f}")

    return all_metrics

def final_train(data_dir):
    """用全部数据训练最终模型"""
    print(f"\n{'='*40}")
    print("用全部数据训练最终模型")
    print(f"{'='*40}")
    
    data_dir = Path(data_dir).resolve()
    yaml_path = None
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as temp_yaml:
            temp_yaml.write(f"""
train: {data_dir / 'new_train'}
val: {data_dir / 'new_train'}
nc: 1
names: ['sheep']
""".strip())
            yaml_path = temp_yaml.name
        
        model = YOLO(r'/home/z5470186/Desktop/yolov12/ultralytics/cfg/models/v12/yolov12s.yaml')
        results = model.train(
            data=yaml_path,
            epochs=100,
            batch=6,
            imgsz=640,
            scale=0.5,
            mosaic=1,
            mixup=0.0,
            copy_paste=0,
            device="0",
            save=True,
            project="final_sheep_model",
            name="best",
            workers=2
        )
        
        # 关键修复：从results对象获取实际保存路径
        actual_model_path = results.save_dir / 'weights' / 'best.pt'
        print(f"最终模型保存到: {actual_model_path}")
        return str(actual_model_path)  # 返回实际路径
        
    finally:
        if yaml_path and os.path.exists(yaml_path):
            os.remove(yaml_path)

def test_final_model(data_dir, model_path):
    """在测试集上评估"""
    data_dir = Path(data_dir).resolve()
    test_dir = data_dir / 'test'
    
    if not test_dir.exists():
        print("没有找到test目录，跳过测试")
        return
    
    print(f"\n{'='*40}")
    print("测试集最终评估")
    print(f"{'='*40}")
    
    yaml_path = None
    try:
        # 创建临时文件但不自动删除
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as temp_yaml:
            temp_yaml.write(f"""
train: {test_dir}
val: {test_dir}
nc: 1
names: ['sheep']
""".strip())
            yaml_path = temp_yaml.name
        
        # 文件已关闭但仍存在，可以安全使用
        model = YOLO(model_path)
        metrics = model.val(data=yaml_path, verbose=True, workers=2)
        
        # 简单直接：都用[0]取第一个元素
        print(f"测试集 F1 score     : {metrics.box.f1[0]:.4f}")
        print(f"测试集 Precision    : {metrics.box.mp:.4f}")
        print(f"测试集 Recall       : {metrics.box.mr:.4f}")
        print(f"测试集 mAP@0.5      : {metrics.box.map50:.4f}")
        print(f"测试集 mAP@0.5:0.95 : {metrics.box.map:.4f}")
    finally:
        # 手动删除临时文件
        if yaml_path and os.path.exists(yaml_path):
            os.remove(yaml_path)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    # 阶段1：合并数据 (为K折验证准备)
    print("阶段1：合并train和valid数据")
    merge_files('aerial_sheep')  # 修改：使用aerial_sheep路径
    
    # 阶段2：K折交叉验证评估稳定性
    print("\n阶段2：K折交叉验证评估稳定性") 
    k_fold_train('aerial_sheep', k=5, seed=42)  # 修改：使用aerial_sheep路径
    
    # 阶段3：用全部数据训练最终模型 (用同样的new_train数据)
    print("\n阶段3：用全部数据训练最终模型")
    model_path = final_train('aerial_sheep')  # 修改：使用aerial_sheep路径
    
    # 阶段4：测试集评估 
    print("\n阶段4：测试集最终评估")
    test_final_model('aerial_sheep', model_path)  # 修改：使用aerial_sheep路径
    
    print("\n完成！")