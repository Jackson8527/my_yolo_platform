import os
import cv2
import subprocess
from ultralytics import YOLO
from config import Config

# 全局变量存储当前加载的模型，避免每次请求都重新加载
current_model_instance = None
current_model_name = None

def get_available_models():
    """ 扫描系统中所有可用的 .pt 模型 """
    models = []
    
    # 1. 扫描根目录下的预训练模型 (yolo11n.pt, yolo11l.pt 等)
    for file in os.listdir(Config.BASE_DIR):
        if file.endswith('.pt'):
            models.append({'name': file, 'path': file, 'type': 'Pretrained'})
            
    # 2. 扫描 runs/ 目录下训练生成的 best.pt
    if os.path.exists(Config.RUNS_FOLDER):
        for root, dirs, files in os.walk(Config.RUNS_FOLDER):
            for file in files:
                if file == 'best.pt':
                    # 获取任务名 (上一级目录名)
                    task_name = os.path.basename(os.path.dirname(os.path.dirname(root)))
                    rel_path = os.path.relpath(os.path.join(root, file), Config.BASE_DIR)
                    models.append({
                        'name': f"{task_name} (best.pt)", 
                        'path': rel_path,
                        'type': 'Trained'
                    })
    return models

def load_model(model_path):
    """ 智能加载模型 (如果已经加载过就不重新加载) """
    global current_model_instance, current_model_name
    
    if current_model_instance is None or current_model_name != model_path:
        print(f"🔄 切换模型为: {model_path}")
        current_model_instance = YOLO(model_path)
        current_model_name = model_path
    
    return current_model_instance

# ... convert_to_h264 保持不变 ...
def convert_to_h264(input_path):
    output_path = input_path.rsplit('.', 1)[0] + "_web.mp4"
    cmd = ["ffmpeg", "-y", "-i", input_path, "-c:v", "libx264", "-preset", "fast",
           "-crf", "23", "-c:a", "aac", "-movflags", "+faststart", output_path]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path

def process_media(input_path, filename, model_path, conf_thres=0.25):
    """ 统一处理图片和视频，接收 model_path 和 conf 参数 """
    model = load_model(model_path)
    ext = os.path.splitext(filename)[1].lower()
    
    # === 图片处理 ===
    if ext in ['.jpg', '.jpeg', '.png']:
        results = model.predict(source=input_path, save=False, conf=conf_thres)
        result_obj = results[0]
        
        detections = []
        for box in result_obj.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = result_obj.names[cls_id]
            detections.append({
                "class": class_name,
                "conf": f"{conf * 100:.1f}",
                "conf_float": conf * 100
            })

        annotated = result_obj.plot()
        result_filename = f"result_{filename}"
        result_path = os.path.join(Config.RESULT_FOLDER, result_filename)
        cv2.imwrite(result_path, annotated)
        
        return "results/" + result_filename, detections, False

    # === 视频处理 ===
    else:
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        tmp_path = os.path.join(Config.RESULT_FOLDER, f"tmp_{filename}")
        out = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        stats = {}
        while True:
            ret, frame = cap.read()
            if not ret: break
            res = model.predict(frame, verbose=False, conf=conf_thres)[0]
            
            for box in res.boxes:
                name = res.names[int(box.cls[0])]
                if name not in stats: stats[name] = 0
                stats[name] += 1 # 简单计数
            
            out.write(res.plot())
        
        cap.release()
        out.release()
        
        # 视频转码
        web_path = convert_to_h264(tmp_path)
        if os.path.exists(tmp_path): os.remove(tmp_path)
        
        # 格式化统计数据
        detections = [{"class": k, "conf": "N/A", "conf_float": 100} for k in stats.keys()]
        
        return "results/" + os.path.basename(web_path), detections, True