from flask import Blueprint, request, render_template
from werkzeug.utils import secure_filename
import os
from config import Config
from services import inference_service

inference_bp = Blueprint('inference', __name__)

@inference_bp.route('/')
def index():
    # 获取所有可用模型传给前端
    models = inference_service.get_available_models()
    return render_template('inference.html', result=None, active_page='inference', models=models)

@inference_bp.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files: return "No file", 400
    file = request.files['file']
    if file.filename == '': return "No file", 400

    # 获取前端传来的参数
    selected_model = request.form.get('model_path', 'yolo11l.pt') # 默认值
    conf_thres = float(request.form.get('conf', 0.25))

    filename = secure_filename(file.filename)
    input_path = os.path.join(Config.UPLOAD_FOLDER, filename)
    file.save(input_path)
    
    # 调用 Service
    result_url, detections, is_video = inference_service.process_media(
        input_path, filename, selected_model, conf_thres
    )

    # 重新获取模型列表(保持下拉框状态)
    models = inference_service.get_available_models()

    return render_template('inference.html', 
                           original=filename, 
                           result=result_url, 
                           detections=detections, 
                           is_video=is_video,
                           active_page='inference',
                           models=models,
                           current_model=selected_model) # 记住刚才选的模型