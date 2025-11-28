from flask import Blueprint, request, jsonify, render_template
from services import training_service

train_bp = Blueprint('train', __name__)

@train_bp.route('/train')
def train_page():
    return render_template('train.html', active_page='train')

@train_bp.route('/start_training', methods=['POST'])
def start_training():
    try:
        # 基础参数
        epochs = request.form.get('epochs')
        batch = request.form.get('batch')
        imgsz = request.form.get('imgsz')
        model_name = request.form.get('model_name')
        project_name = request.form.get('project_name')
        file = request.files.get('dataset')

        # === 收集所有额外参数 (增强 + 系统 + Resume) ===
        extra_args = {
            # 系统
            "resume": request.form.get('resume'), # True/False
            "device": request.form.get('device'),
            "workers": request.form.get('workers'),
            "patience": request.form.get('patience'),
            "optimizer": request.form.get('optimizer'),
            "cos_lr": request.form.get('cos_lr'),
            
            # 数据增强 (Augmentation)
            "mosaic": request.form.get('mosaic'),
            "mixup": request.form.get('mixup'),
            "degrees": request.form.get('degrees'), # 旋转
            "flipud": request.form.get('flipud'),   # 上下翻转
            "fliplr": request.form.get('fliplr'),   # 左右翻转
        }

        training_service.start_training_task(
            file, model_name, epochs, batch, imgsz, project_name, extra_args
        )
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# === 新增：获取训练过程中的验证图片 ===
@train_bp.route('/get_val_image')
def get_val_image():
    project_name = request.args.get('project_name')
    if not project_name: return "", 404
    
    # YOLO 训练时会生成 val_batch0_labels.jpg 或 val_batch0_pred.jpg
    # 我们优先显示 pred (预测结果)，如果没有则显示 labels (标注真值)
    base_path = os.path.join(Config.RUNS_FOLDER, project_name)
    
    pred_img = os.path.join(base_path, 'val_batch0_pred.jpg')
    label_img = os.path.join(base_path, 'val_batch0_labels.jpg')
    
    if os.path.exists(pred_img):
        return send_file(pred_img, mimetype='image/jpeg')
    elif os.path.exists(label_img):
        return send_file(label_img, mimetype='image/jpeg')
    else:
        return "", 404 # 图片还没生成

@train_bp.route('/stop_training', methods=['POST'])
def stop_training():
    if training_service.stop_training():
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 400

@train_bp.route('/get_logs')
def get_logs():
    logs, is_training = training_service.get_logs()
    return jsonify({"logs": logs, "is_training": is_training})

# === 新增：获取图表数据 ===
@train_bp.route('/get_metrics')
def get_metrics():
    project_name = request.args.get('project_name')
    if not project_name: return jsonify({})
    data = training_service.get_training_metrics(project_name)
    if data: return jsonify({"status": "success", "data": data})
    return jsonify({"status": "waiting"})

# === 新增：获取进度条数据 ===
@train_bp.route('/get_progress')
def get_progress():
    project_name = request.args.get('project_name')
    if not project_name: return jsonify({})
    data = training_service.get_latest_metrics(project_name)
    return jsonify(data if data else {})