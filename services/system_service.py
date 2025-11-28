import psutil
import pynvml

def get_system_status():
    status = {
        "cpu_percent": 0,
        "ram_percent": 0,
        "gpu_name": "N/A",
        "gpu_util": 0,
        "gpu_mem": 0,
        "gpu_temp": 0
    }

    # 1. CPU & RAM
    try:
        status["cpu_percent"] = psutil.cpu_percent(interval=None)
        status["ram_percent"] = psutil.virtual_memory().percent
    except:
        pass

    # 2. GPU (NVIDIA)
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 默认获取第0张卡
        
        name = pynvml.nvmlDeviceGetName(handle)
        # 兼容旧版 pynvml 返回 bytes 的情况
        if isinstance(name, bytes):
            name = name.decode('utf-8')
            
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

        status["gpu_name"] = name
        status["gpu_util"] = util.gpu
        status["gpu_mem"] = int((mem.used / mem.total) * 100)
        status["gpu_temp"] = temp
        
    except Exception:
        # 如果没有 NVIDIA 显卡或驱动未安装，保持默认值
        pass

    return status