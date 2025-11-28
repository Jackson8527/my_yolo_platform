from flask import Blueprint, render_template, jsonify, request
from services import dashboard_service, system_service

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/')  # 根路径现在是 Dashboard
@dashboard_bp.route('/dashboard')
def index():
    return render_template('dashboard.html', active_page='dashboard')

@dashboard_bp.route('/settings')
def settings():
    # 获取训练历史列表，用于在设置页管理（删除）
    runs = dashboard_service.get_training_history()
    return render_template('settings.html', active_page='settings', runs=runs)

@dashboard_bp.route('/api/dashboard_stats')
def get_stats():
    # 获取静态统计
    stats = dashboard_service.get_global_stats()
    # 获取实时硬件信息
    sys_status = system_service.get_system_status()
    return jsonify({**stats, **sys_status})

@dashboard_bp.route('/api/clear_cache', methods=['POST'])
def clear_cache():
    count = dashboard_service.clear_cache_files()
    return jsonify({"status": "success", "count": count})

@dashboard_bp.route('/api/delete_run', methods=['POST'])
def delete_run():
    run_name = request.json.get('name')
    if dashboard_service.delete_run(run_name):
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 400