from flask import Blueprint, render_template, request, jsonify, send_from_directory
import os
from config import Config
from services import labeling_service
from flask import send_file

label_bp = Blueprint('label', __name__)

# 配置一个特殊的静态路由，用于访问 raw_images 文件夹里的图片
@label_bp.route('/static_raw/<path:filename>')
def serve_raw_image(filename):
    return send_from_directory(labeling_service.RAW_IMAGES_DIR, filename)

@label_bp.route('/labeling')
def index():
    return render_template('labeling.html', active_page='labeling')

@label_bp.route('/api/get_images')
def get_images():
    images = labeling_service.get_images_list()
    return jsonify(images)

@label_bp.route('/api/save_label', methods=['POST'])
def save_label():
    data = request.json
    filename = data.get('filename')
    boxes = data.get('boxes') # List of {x, y, w, h, class_id}
    classes = data.get('classes') # List of strings
    
    if labeling_service.save_annotation(filename, boxes, classes):
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 500

@label_bp.route('/api/upload_raw', methods=['POST'])
def upload_raw():
    # 简单的图片上传接口
    if 'file' not in request.files: return "No file", 400
    file = request.files['file']
    if file.filename == '': return "Empty", 400
    
    path = os.path.join(labeling_service.RAW_IMAGES_DIR, file.filename)
    file.save(path)
    return jsonify({"status": "success"})

@label_bp.route('/api/export_dataset')
def export_dataset():
    try:
        zip_path = labeling_service.export_dataset_to_zip()
        return send_file(zip_path, as_attachment=True, download_name='my_yolo_dataset.zip')
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400