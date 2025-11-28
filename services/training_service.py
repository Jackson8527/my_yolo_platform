import os
import threading
import subprocess
import shutil
import zipfile
import sys
import json
import yaml
import time
import pandas as pd
from config import Config

class TrainingState:
    def __init__(self):
        self.is_training = False
        self.logs = []
        self.process = None
        self.stop_event = False 

state = TrainingState()

# ================= COCO 格式转换器 =================
class COCOConverter:
    @staticmethod
    def convert(dataset_root):
        """ 自动检测并转换 COCO 格式 """
        train_json, val_json = None, None
        img_train_dir, img_val_dir = None, None
        
        # 1. 扫描文件
        for root, dirs, files in os.walk(dataset_root):
            if 'train2017.json' in files: train_json = os.path.join(root, 'train2017.json')
            if 'val2017.json' in files: val_json = os.path.join(root, 'val2017.json')
            if 'train2017' in dirs: img_train_dir = os.path.join(root, 'train2017')
            if 'val2017' in dirs: img_val_dir = os.path.join(root, 'val2017')

        # 如果找不到 json，尝试直接找 data.yaml
        if not train_json:
            for root, _, files in os.walk(dataset_root):
                if 'data.yaml' in files: return os.path.join(root, 'data.yaml')
            raise Exception("未找到 train2017.json 或 data.yaml")

        state.logs.append(f"检测到 COCO 格式，正在转换...\n")
        
        # 2. 创建目录
        output_dir = os.path.join(dataset_root, 'yolo_formatted')
        for split in ['train', 'val']:
            os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
            os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

        # 3. 转换
        names = COCOConverter._process_json(train_json, img_train_dir, output_dir, 'train')
        if val_json and img_val_dir:
            COCOConverter._process_json(val_json, img_val_dir, output_dir, 'val')

        # 4. 生成 yaml
        yaml_content = {
            'path': output_dir,
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(names),
            'names': names
        }
        yaml_path = os.path.join(output_dir, 'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False)
            
        state.logs.append(f"✅ 转换完成！类别: {names}\n")
        return yaml_path

    @staticmethod
    def _process_json(json_path, img_source, output_base, split):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        cat_map = {cat['id']: i for i, cat in enumerate(data['categories'])}
        names = [cat['name'] for cat in data['categories']]
        images_info = {img['id']: img for img in data['images']}
        
        # 复制图片
        for img_id, info in images_info.items():
            src = os.path.join(img_source, info['file_name'])
            dst = os.path.join(output_base, 'images', split, info['file_name'])
            if os.path.exists(src): shutil.copy(src, dst)

        # 生成标注
        for ann in data['annotations']:
            img = images_info.get(ann['image_id'])
            if not img: continue
            
            x, y, w, h = ann['bbox']
            # 归一化 xywh
            dw = 1. / img['width']
            dh = 1. / img['height']
            x_center = (x + w / 2.0) * dw
            y_center = (y + h / 2.0) * dh
            w = w * dw
            h = h * dh
            
            cls_id = cat_map[ann['category_id']]
            txt_name = os.path.splitext(img['file_name'])[0] + ".txt"
            txt_path = os.path.join(output_base, 'labels', split, txt_name)
            
            with open(txt_path, 'a') as f:
                f.write(f"{cls_id} {x_center} {y_center} {w} {h}\n")
        
        return names

# ================= 数据读取逻辑 =================

def get_training_metrics(project_name):
    """ 读取 results.csv 返回所有数据用于画图 """
    csv_path = os.path.join(Config.RUNS_FOLDER, project_name, 'results.csv')
    if not os.path.exists(csv_path): return None
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns] # 去空格
        return {
            "epoch": df['epoch'].tolist(),
            "box_loss": df['train/box_loss'].tolist(),
            "map50": df['metrics/mAP50(B)'].tolist()
        }
    except: return None

def get_latest_metrics(project_name):
    """ 读取最后一行数据用于进度条 """
    csv_path = os.path.join(Config.RUNS_FOLDER, project_name, 'results.csv')
    if not os.path.exists(csv_path): return None
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        if df.empty: return None
        
        last = df.iloc[-1]
        return {
            "epoch": int(last['epoch']),
            "box_loss": round(last['train/box_loss'], 5),
            "cls_loss": round(last.get('train/cls_loss', 0), 5),
            "map50": round(last['metrics/mAP50(B)'], 3)
        }
    except: return None

# ================= 训练线程逻辑 =================

def _run_full_process_thread(zip_path, dataset_name, model_name, epochs, batch, imgsz, project_name, extra_args):
    global state
    state.stop_event = False
    
    try:
        # === 1. 判断是否为恢复训练 (Resume) ===
        is_resume = extra_args.get('resume') == 'True'
        resume_path = os.path.join(Config.RUNS_FOLDER, project_name, 'weights', 'last.pt')
        
        yaml_path = None # 初始化

        if is_resume:
            if not os.path.exists(resume_path):
                state.logs.append(f"❌ 无法恢复训练：未找到 {resume_path}\n")
                state.is_training = False
                return
            state.logs.append(f"🔄 [1/3] 检测到恢复训练请求，加载: {resume_path}...\n")
            # 恢复训练时，不需要解压数据集（假设已经存在），直接复用
            # 但为了保险，我们还是定义一下 yaml 路径，防止 yolo 找不到
            # 这里简单处理：假设用户之前的路径没变。
            # 实际上 resume=True 时，YOLO 会从 last.pt 里读取所有配置，我们可以跳过解压步骤
        else:
            # === 非恢复训练：正常解压和转换 ===
            extract_path = os.path.join(Config.DATASET_FOLDER, dataset_name)
            state.logs.append(f"📦 [1/3] 解压数据集: {dataset_name}...\n")
            
            if os.path.exists(extract_path): shutil.rmtree(extract_path)
            os.makedirs(extract_path)
            
            with zipfile.ZipFile(zip_path, 'r') as z: z.extractall(extract_path)
            if state.stop_event: raise Exception("任务被终止")

            state.logs.append(f"🔄 [2/3] 检查格式...\n")
            try:
                # 假设你保留了 COCOConverter
                from services.training_service import COCOConverter
                yaml_path = COCOConverter.convert(extract_path)
            except Exception as e:
                # 兜底寻找
                found = False
                for r, _, f in os.walk(extract_path):
                    if 'data.yaml' in f:
                        yaml_path = os.path.join(r, 'data.yaml')
                        found = True
                        break
                if not found: raise Exception("找不到 data.yaml 且无法自动转换")
            
            state.logs.append(f"✅ 数据集准备就绪: {yaml_path}\n")

        if state.stop_event: raise Exception("任务被终止")

        # === 3. 构造训练命令 ===
        state.logs.append(f"🚀 [3/3] 启动训练...\n")
        
        # 寻找 yolo 执行路径
        yolo_exe = os.path.join(os.path.dirname(sys.executable), 'yolo')
        if not os.path.exists(yolo_exe): yolo_exe = 'yolo'
        if os.name == 'nt':
            win_exe = os.path.join(os.path.dirname(sys.executable), 'Scripts', 'yolo.exe')
            if os.path.exists(win_exe): yolo_exe = win_exe

        cmd = [yolo_exe, "train"]

        if is_resume:
            # 恢复训练核心参数
            cmd.append(f"model={resume_path}")
            cmd.append("resume=True")
        else:
            # 新训练核心参数
            cmd.append(f"model={model_name}")
            cmd.append(f"data={yaml_path}")
            cmd.append(f"epochs={epochs}")
            cmd.append(f"batch={batch}")
            cmd.append(f"imgsz={imgsz}")
            cmd.append(f"project={Config.RUNS_FOLDER}")
            cmd.append(f"name={project_name}")
            cmd.append("exist_ok=True")
            
            # === 添加增强参数 (Augmentation) ===
            # 只有在新训练时生效，恢复训练会沿用之前的设置
            aug_params = ['degrees', 'translate', 'scale', 'shear', 'perspective', 'flipud', 'fliplr', 'mosaic', 'mixup']
            for arg in aug_params:
                if extra_args.get(arg):
                    cmd.append(f"{arg}={extra_args.get(arg)}")

            # 添加系统参数
            sys_params = ['device', 'workers', 'patience', 'optimizer', 'cos_lr']
            for arg in sys_params:
                if extra_args.get(arg):
                    cmd.append(f"{arg}={extra_args.get(arg)}")

        state.logs.append(f"🔧 命令: {' '.join(cmd)}\n")
        
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
            text=True, bufsize=1, encoding='utf-8', errors='replace'
        )
        state.process = process

        for line in iter(process.stdout.readline, ''):
            if state.stop_event:
                process.terminate()
                state.logs.append("\n🛑 用户点击终止，正在停止...\n")
                break
            if line: state.logs.append(line)
        
        if not state.stop_event:
            if process.wait() == 0: state.logs.append("\n✅ 训练完成！\n")
            else: state.logs.append("\n❌ 训练异常退出\n")

    except Exception as e:
        state.logs.append(f"\n❌ 错误: {str(e)}\n")
    finally:
        state.is_training = False
        state.process = None

def start_training_task(file, model_name, epochs, batch, imgsz, project_name, extra_args):
    if state.is_training: raise Exception("已有任务在运行")
    
    # 如果是 Resume，不需要上传文件
    is_resume = extra_args.get('resume') == 'True'
    
    zip_path = ""
    dataset_name = ""

    if not is_resume:
        if not file: raise Exception("新训练必须上传数据集")
        filename = file.filename
        zip_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(zip_path)
        dataset_name = os.path.splitext(filename)[0]
    
    state.logs = [f"--- 开始任务: {project_name} {'(恢复训练)' if is_resume else ''} ---\n"]
    state.is_training = True
    
    thread = threading.Thread(
        target=_run_full_process_thread,
        args=(zip_path, dataset_name, model_name, epochs, batch, imgsz, project_name, extra_args)
    )
    thread.daemon = True
    thread.start()

def stop_training():
    if state.is_training:
        state.stop_event = True
        if state.process: state.process.terminate()
        return True
    return False

def get_logs():
    return "".join(state.logs), state.is_training