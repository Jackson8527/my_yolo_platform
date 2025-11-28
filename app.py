from flask import Flask, jsonify
from config import Config
from routes.inference_routes import inference_bp
from routes.training_routes import train_bp
from services import system_service # 导入硬件监控服务
from routes.labeling_routes import label_bp
from routes.dashboard_routes import dashboard_bp

def create_app():
    app = Flask(__name__)
    
    # 1. 初始化
    Config.init_dirs()
    
    # 2. 注册蓝图
    app.register_blueprint(inference_bp)
    app.register_blueprint(train_bp)
    app.register_blueprint(label_bp)
    app.register_blueprint(dashboard_bp)    
    
    # 3. 注册系统监控路由 (直接写在这里最方便)
    @app.route('/system_status')
    def system_status():
        return jsonify(system_service.get_system_status())
    
    return app

if __name__ == '__main__':
    app = create_app()
    print("🚀 YOLO 平台已启动: http://localhost:7860")
    app.run(host='0.0.0.0', port=7860, debug=True)