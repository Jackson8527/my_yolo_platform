# config.py
import os

class Config:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static/uploads')
    RESULT_FOLDER = os.path.join(BASE_DIR, 'static/results')
    DATASET_FOLDER = os.path.join(BASE_DIR, 'datasets')
    RUNS_FOLDER = os.path.join(BASE_DIR, 'runs')
    
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'zip'}
    
    # 确保目录存在
    @staticmethod
    def init_dirs():
        for folder in [Config.UPLOAD_FOLDER, Config.RESULT_FOLDER, 
                       Config.DATASET_FOLDER, Config.RUNS_FOLDER]:
            os.makedirs(folder, exist_ok=True)

    @staticmethod
    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS