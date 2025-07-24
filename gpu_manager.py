# src/gpu/gpu_manager.py
import torch
import cupy as cp
import rapids.ai as rapids

class GPUManager:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_available = torch.cuda.is_available()
        
    def get_gpu_info(self):
        if self.gpu_available:
            return {
                'name': torch.cuda.get_device_name(0),
                'memory_total': torch.cuda.get_device_properties(0).total_memory,
                'memory_free': torch.cuda.mem_get_info()[0],
                'compute_capability': torch.cuda.get_device_capability(0)
            }
        return None