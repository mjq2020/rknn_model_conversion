import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum


@dataclass
class ServerConfig:
    """Server configuration"""

    host: str = "0.0.0.0"
    port: int = 8080
    max_workers: int = 4
    upload_folder: str = "./uploads"
    output_folder: str = "./outputs"
    temp_folder: str = "./temp"
    max_file_size: int = 500 * 1024 * 1024  # 500MB
    allowed_extensions: set = field(
        default_factory=lambda: {
            ".onnx",
            ".tflite",
            ".prototxt",
            ".caffemodel",
            ".pytorch",
            ".pth",
            ".pt",
            ".pb",
            ".ckpt",
            ".meta",
            ".data",
            ".index",
            ".cfg",
            ".weights",
            ".darknet",
            ".data-00000-of-00001",
        }
    )


@dataclass
class ModelFiles:
    """Model file group"""

    primary_file: str  # Primary file path
    secondary_files: List[str] = field(default_factory=list)  # Auxiliary file paths
    additional_files: List[str] = field(default_factory=list)  # Additional file paths
    model_type: str = ""  # Model type

    def get_all_files(self) -> List[str]:
        """Get all file paths"""
        return [self.primary_file] + self.secondary_files

    def get_model_name(self) -> str:
        """Extract model name from primary file path"""
        return os.path.splitext(os.path.basename(self.primary_file))[0]


class ModelType(Enum):
    ONNX = ["onnx"]
    TFLITE = ["tflite"]
    CAFFE = ["prototxt", "caffemodel"]
    PYTORCH = ["pt", "pth", "pytorch"]
    TENSORFLOW = ["pb", "index", "data-00000-of-00001"]
    DARKNET = ["cfg", "weights"]
    ALL = [
        "onnx",
        "tflite",
        "prototxt",
        "caffemodel",
        "pt",
        "pth",
        "pytorch",
        "pb",
        "index",
        "data-00000-of-00001",
        "cfg",
        "weights",
    ]
    UNKNOWN = ["unknown"]


@dataclass
class RKNNConverterConfig:
    # config for rknn.config
    mean_values: List[float] = field(default_factory=lambda: [0, 0, 0])
    std_values: List[float] = field(default_factory=lambda: [255, 255, 255])
    quantized_dtype: str = "w8a8"
    quantized_algorithm: str = "normal"
    quantized_method: str = "channel"
    quantized_hybrid_level: int = 0
    target_platform: Optional[str] = "rk3588"
    quant_img_RGB2BGR: bool = False
    float_dtype: str = "float16"
    optimization_level: int = 3
    custom_string: Optional[str] = None
    remove_weight: bool = False
    compress_weight: bool = False
    inputs_yuv_fmt: Optional[str] = None
    single_core_mode: bool = False
    dynamic_input: Optional[List[List[Tuple[int, int, int]]]] = None
    model_pruning: bool = False
    op_target: Optional[Dict[str, str]] = None
    quantize_weight: bool = False
    remove_reshape: bool = False
    sparse_infer: bool = False
    enable_flash_attention: bool = False
    auto_hybrid_cos_thresh: float = 0.98
    auto_hybrid_euc_thresh: Optional[float] = None

    # build config
    do_quantization: bool = True
    dataset: str = "./images.txt"
    rknn_batch_size: Optional[int] = None
    auto_hybrid: bool = False

    # torch config
    input_size_list: List[List] = field(default_factory=lambda: [[1, 3, 224, 224]])

    def config(self) -> dict:
        return {
            "mean_values": self.mean_values,
            "std_values": self.std_values,
            "quantized_dtype": self.quantized_dtype,
            "quantized_algorithm": self.quantized_algorithm,
            "quantized_method": self.quantized_method,
            "quantized_hybrid_level": self.quantized_hybrid_level,
            "target_platform": self.target_platform,
            "quant_img_RGB2BGR": self.quant_img_RGB2BGR,
            "float_dtype": self.float_dtype,
            "optimization_level": self.optimization_level,
            "custom_string": self.custom_string,
            "remove_weight": self.remove_weight,
            "compress_weight": self.compress_weight,
            "inputs_yuv_fmt": self.inputs_yuv_fmt,
            "single_core_mode": self.single_core_mode,
            "dynamic_input": self.dynamic_input,
            "model_pruning": self.model_pruning,
            "op_target": self.op_target,
            "quantize_weight": self.quantize_weight,
            "remove_reshape": self.remove_reshape,
            "sparse_infer": self.sparse_infer,
            "enable_flash_attention": self.enable_flash_attention,
            "auto_hybrid_cos_thresh": self.auto_hybrid_cos_thresh,
            "auto_hybrid_euc_thresh": self.auto_hybrid_euc_thresh,
        }

    def build_config(self) -> dict:
        return {
            "do_quantization": self.do_quantization,
            "dataset": self.dataset,
            "rknn_batch_size": self.rknn_batch_size,
            "auto_hybrid": self.auto_hybrid,
            # "quant_img_RGB2BGR": self.quant_img_RGB2BGR,
            # "float_dtype": self.float_dtype
        }

    def torch_config(self) -> dict:
        return {"input_size_list": self.input_size_list}

    def update_config(self, config: dict):
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass
class ConversionTask:
    """Conversion task configuration"""

    task_id: str
    model_files: ModelFiles  # Changed to model file group
    config: RKNNConverterConfig
    output_path: Optional[str] = None
    callback_url: Optional[str] = None
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def model_path(self) -> str:
        """Compatibility property, returns primary file path"""
        return self.model_files.primary_file

    def get_output_path(self, output_folder: str = "./outputs") -> str:
        """Generate output file path"""
        if self.output_path:
            return self.output_path

        # Extract filename from model file group and replace extension with .rknn
        model_name = self.model_files.get_model_name()
        output_filename = f"{model_name}_{self.task_id}.rknn"
        return os.path.join(output_folder, output_filename)

    def get_dataset_path(self) -> str:
        """Get dataset path"""
        # Use dataset path from config, or default value if not available
        if hasattr(self.config, "dataset") and self.config.dataset:
            return self.config.dataset
        return "./images.txt"


# Default configurations
DEFAULT_SERVER_CONFIG = ServerConfig()
DEFAULT_CONVERTER_CONFIG = RKNNConverterConfig()


# Ensure necessary directories exist
def ensure_directories():
    """Ensure necessary directories exist"""
    config = DEFAULT_SERVER_CONFIG
    for folder in [config.upload_folder, config.output_folder, config.temp_folder]:
        os.makedirs(folder, exist_ok=True)
