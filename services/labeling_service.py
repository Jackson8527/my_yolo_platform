import os
from config import Config
import shutil
import random
import yaml
import zipfile

# 定义标注数据的存放路径
RAW_IMAGES_DIR = os.path.join(Config.BASE_DIR, 'datasets', 'raw_images')
LABELS_OUTPUT_DIR = os.path.join(Config.BASE_DIR, 'datasets', 'labels')

# 确保目录存在
os.makedirs(RAW_IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_OUTPUT_DIR, exist_ok=True)

def get_images_list():
    """获取所有待标注图片"""
    images = []
    if not os.path.exists(RAW_IMAGES_DIR): return []
    
    # 支持的图片格式
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for f in os.listdir(RAW_IMAGES_DIR):
        ext = os.path.splitext(f)[1].lower()
        if ext in valid_exts:
            # 检查是否已经标注过 (是否存在同名 txt)
            txt_name = os.path.splitext(f)[0] + ".txt"
            is_labeled = os.path.exists(os.path.join(LABELS_OUTPUT_DIR, txt_name))
            
            images.append({
                "name": f,
                "url": f"/static_raw/{f}", # 我们需要注册一个静态路由
                "is_labeled": is_labeled
            })
    
    # 排序：未标注的排前面
    images.sort(key=lambda x: x['is_labeled'])
    return images

def save_annotation(filename, boxes, classes):
    """
    保存标注结果为 YOLO 格式 txt
    boxes: [{x, y, w, h, class_id}, ...] (归一化后的数据)
    """
    txt_name = os.path.splitext(filename)[0] + ".txt"
    txt_path = os.path.join(LABELS_OUTPUT_DIR, txt_name)
    
    with open(txt_path, 'w') as f:
        for box in boxes:
            # YOLO format: class_id center_x center_y width height
            line = f"{box['class_id']} {box['x']} {box['y']} {box['w']} {box['h']}\n"
            f.write(line)
            
    # 同时保存 classes.txt 以便记录类别名称
    classes_path = os.path.join(LABELS_OUTPUT_DIR, 'classes.txt')
    # 只有当文件不存在或类别更新时才写（这里简化处理，覆盖写入）
    with open(classes_path, 'w') as f:
        for name in classes:
            f.write(f"{name}\n")
            
    return True

def get_existing_labels(filename):
    """读取已有的标注（如果有）"""
    txt_name = os.path.splitext(filename)[0] + ".txt"
    txt_path = os.path.join(LABELS_OUTPUT_DIR, txt_name)
    
    boxes = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    boxes.append({
                        "class_id": int(parts[0]),
                        "x": float(parts[1]),
                        "y": float(parts[2]),
                        "w": float(parts[3]),
                        "h": float(parts[4])
                    })
    return boxes

def export_dataset_to_zip(val_split=0.2):
    """
    将标注好的数据打包成 YOLO 训练所需的 Zip 格式
    val_split: 验证集比例 (默认 20%)
    """
    # 1. 准备临时目录
    export_dir = os.path.join(Config.BASE_DIR, 'temp_export')
    if os.path.exists(export_dir): shutil.rmtree(export_dir)
    
    # 创建 YOLO 标准目录结构
    sub_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for d in sub_dirs:
        os.makedirs(os.path.join(export_dir, d), exist_ok=True)
        
    # 2. 获取所有已标注的图片
    valid_pairs = [] # 存放 (图片路径, 标签路径)
    
    images = os.listdir(RAW_IMAGES_DIR)
    for img_name in images:
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')): continue
        
        # 找对应的 txt
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        txt_path = os.path.join(LABELS_OUTPUT_DIR, txt_name)
        img_path = os.path.join(RAW_IMAGES_DIR, img_name)
        
        if os.path.exists(txt_path):
            valid_pairs.append((img_path, txt_path, img_name, txt_name))
            
    if not valid_pairs:
        raise Exception("没有找到已标注的数据！请先进行标注。")

    # 3. 随机打乱并切分
    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * (1 - val_split))
    train_set = valid_pairs[:split_idx]
    val_set = valid_pairs[split_idx:]
    
    # 4. 复制文件
    def copy_files(dataset, split_name):
        for img_p, txt_p, img_n, txt_n in dataset:
            shutil.copy(img_p, os.path.join(export_dir, 'images', split_name, img_n))
            shutil.copy(txt_p, os.path.join(export_dir, 'labels', split_name, txt_n))
            
    copy_files(train_set, 'train')
    copy_files(val_set, 'val')
    
    # 5. 生成 data.yaml
    classes_path = os.path.join(LABELS_OUTPUT_DIR, 'classes.txt')
    names = []
    if os.path.exists(classes_path):
        with open(classes_path, 'r') as f:
            names = [line.strip() for line in f.readlines() if line.strip()]
            
    yaml_content = {
        'path': '../datasets/my_dataset', # 这里的路径在训练解压时会被覆盖，写个相对的即可
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(names),
        'names': names
    }
    
    yaml_path = os.path.join(export_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    # 6. 打包成 Zip
    zip_filename = 'yolo_dataset_export.zip'
    zip_path = os.path.join(Config.BASE_DIR, zip_filename)
    
    with zipfile.ZipFile(zip_path, 'w') as z:
        for root, dirs, files in os.walk(export_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                # 保持压缩包内的相对路径
                rel_path = os.path.relpath(abs_path, export_dir)
                z.write(abs_path, rel_path)
                
    # 清理临时目录
    shutil.rmtree(export_dir)
    
    return zip_path