import os
import shutil
import pandas as pd
from config import Config
from services import system_service

def get_global_stats():
    """获取全局统计信息"""
    stats = {
        "model_count": 0,
        "dataset_count": 0,
        "total_runs": 0,
        "best_map": 0.0,
        "disk_usage": 0
    }

    # 1. 统计数据集 (文件夹数量)
    if os.path.exists(Config.DATASET_FOLDER):
        # 排除非目录
        dirs = [d for d in os.listdir(Config.DATASET_FOLDER) if os.path.isdir(os.path.join(Config.DATASET_FOLDER, d))]
        stats["dataset_count"] = len(dirs)

    # 2. 统计训练任务 & 寻找最佳 mAP
    runs_dir = Config.RUNS_FOLDER
    if os.path.exists(runs_dir):
        runs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
        stats["total_runs"] = len(runs)
        
        # 遍历所有任务，找最高 mAP
        max_map = 0
        for run in runs:
            csv_path = os.path.join(runs_dir, run, 'results.csv')
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    df.columns = [c.strip() for c in df.columns]
                    # 获取该次训练的最大 mAP50
                    current_max = df['metrics/mAP50(B)'].max()
                    if current_max > max_map:
                        max_map = current_max
                except:
                    pass
        stats["best_map"] = round(max_map * 100, 2) # 转百分比

    # 3. 统计模型数量 (.pt 文件)
    # 包括根目录的预训练模型 + runs 里的 best.pt
    base_models = len([f for f in os.listdir(Config.BASE_DIR) if f.endswith('.pt')])
    stats["model_count"] = base_models + stats["total_runs"] # 简单估算

    # 4. 磁盘占用 (static文件夹)
    total_size = 0
    for folder in [Config.UPLOAD_FOLDER, Config.RESULT_FOLDER, Config.RUNS_FOLDER, Config.DATASET_FOLDER]:
        if os.path.exists(folder):
            for dirpath, dirnames, filenames in os.walk(folder):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if not os.path.islink(fp):
                        total_size += os.path.getsize(fp)
    
    stats["disk_usage"] = round(total_size / (1024 * 1024), 1) # MB
    
    return stats

def get_training_history():
    """获取所有训练任务的简报列表"""
    history = []
    runs_dir = Config.RUNS_FOLDER
    if not os.path.exists(runs_dir): return []

    for run_name in os.listdir(runs_dir):
        run_path = os.path.join(runs_dir, run_name)
        if not os.path.isdir(run_path): continue
        
        csv_path = os.path.join(run_path, 'results.csv')
        item = {
            "name": run_name,
            "epochs": 0,
            "last_map": 0,
            "status": "Unknown"
        }
        
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df.columns = [c.strip() for c in df.columns]
                item["epochs"] = len(df)
                item["last_map"] = round(df.iloc[-1]['metrics/mAP50(B)'] * 100, 2)
                item["status"] = "Completed" # 简单判断，有csv就算完成
            except:
                item["status"] = "Error"
        else:
            item["status"] = "No Data"
            
        history.append(item)
    
    return history

def clear_cache_files():
    """清理临时上传和结果文件"""
    cleared_count = 0
    for folder in [Config.UPLOAD_FOLDER, Config.RESULT_FOLDER]:
        if os.path.exists(folder):
            for f in os.listdir(folder):
                file_path = os.path.join(folder, f)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        cleared_count += 1
                except Exception as e:
                    print(e)
    return cleared_count

def delete_run(run_name):
    """删除指定的训练任务"""
    path = os.path.join(Config.RUNS_FOLDER, run_name)
    if os.path.exists(path):
        shutil.rmtree(path)
        return True
    return False