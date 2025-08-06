from rknn.api import RKNN
import os
from utils.config import RKNNConverterConfig, ModelType


class RKNNConverter:
    def __init__(
        self,
        model_files,
        output_path: str,
        dataset_path: str = "./images.txt",
        config: RKNNConverterConfig = RKNNConverterConfig(),
    ):
        """
        Initialize RKNN converter

        Args:
            model_files: ModelFiles object containing primary and auxiliary file paths
            output_path: Output path
            dataset_path: Dataset path
            config: Conversion configuration
        """
        self.rknn_config: RKNNConverterConfig = config
        print(self.rknn_config)
        self.model_files = model_files
        self.model_path = model_files.primary_file  # Compatibility
        self.output_path = output_path
        self.dataset_path = dataset_path
        self.current_model_type = self.check_input_model()
        self.rknn = RKNN()
        self.config()

    def check_input_model(self) -> ModelType:
        """Check input model type"""
        primary_ext = os.path.splitext(self.model_path)[1].lower()
        print("*********", primary_ext)
        if primary_ext == ".onnx":
            return ModelType.ONNX
        elif primary_ext == ".tflite":
            return ModelType.TFLITE
        elif primary_ext == ".prototxt":
            return ModelType.CAFFE
        elif primary_ext in [".pt", ".pth", ".pytorch"]:
            return ModelType.PYTORCH
        elif primary_ext in [".pb", ".index"]:
            return ModelType.TENSORFLOW
        elif primary_ext == ".cfg":
            return ModelType.DARKNET
        elif os.path.isdir(self.model_path):
            # Check if it's a SavedModel directory
            if os.path.exists(os.path.join(self.model_path, "saved_model.pb")):
                return ModelType.TENSORFLOW
            else:
                return ModelType.UNKNOWN
        else:
            return ModelType.UNKNOWN

    def convert(self, progress_callback=None):
        # load model
        try:
            self.load_model()
        except Exception as e:
            print(f"Error: {e}")
            self.rknn.release()
            return False, e
        if progress_callback is not None:
            progress_callback(30)
        # build graph
        try:
            self.rknn.build(**self.rknn_config.build_config())
        except Exception as e:
            print(f"Error: {e}")
            self.rknn.release()
            return False, e
        if progress_callback is not None:
            progress_callback(60)
        # export rknn
        try:
            self.rknn.export_rknn(self.output_path)
        except Exception as e:
            print(f"Error: {e}")
            self.rknn.release()
            return False, e
        if progress_callback is not None:
            progress_callback(90)
        self.rknn.release()

        # Clean up temporary files
        self._cleanup_temp_files()

        if progress_callback is not None:
            progress_callback(100)
        return True, None

    def config(self):
        self.rknn.config(**self.rknn_config.config())

    def _cleanup_temp_files(self):
        """Clean up temporary files created during conversion"""
        # Clean up temporary TFLite files
        if hasattr(self, "_temp_tflite_path") and self._temp_tflite_path:
            if os.path.exists(self._temp_tflite_path):
                try:
                    os.unlink(self._temp_tflite_path)
                    print(f"Cleaned up temporary TFLite file: {self._temp_tflite_path}")
                except Exception as e:
                    print(f"Failed to clean up temporary file: {e}")
                finally:
                    self._temp_tflite_path = None

        # Clean up temporary SavedModel directory
        if hasattr(self, "_temp_savedmodel_dir") and self._temp_savedmodel_dir:
            if os.path.exists(self._temp_savedmodel_dir):
                try:
                    import shutil

                    shutil.rmtree(self._temp_savedmodel_dir)
                    print(
                        f"Cleaned up temporary SavedModel directory: {self._temp_savedmodel_dir}"
                    )
                except Exception as e:
                    print(f"Failed to clean up temporary directory: {e}")
                finally:
                    self._temp_savedmodel_dir = None

    def _save_model2tflite(self, model_path):
        import tensorflow as tf

        tflite_model_path = os.path.join(os.path.dirname(model_path), "model.tflite")
        if os.path.exists(tflite_model_path):
            return tflite_model_path
        converter = tf.lite.TFLiteConverter.from_saved_model(
            os.path.dirname(model_path)
        )
        tflite_model = converter.convert()
        with open(tflite_model_path, "wb") as f:
            f.write(tflite_model)

        return tflite_model_path

    def load_model(self):
        """Load model, supporting multiple file formats"""
        if self.current_model_type == ModelType.ONNX:
            self.rknn.load_onnx(self.model_path)
        elif self.current_model_type == ModelType.TFLITE:
            self.rknn.load_tflite(self.model_path)
        elif self.current_model_type == ModelType.CAFFE:
            # Caffe model requires prototxt and caffemodel files
            prototxt_path = self.model_path  # Primary file is prototxt
            caffemodel_path = None

            # Find corresponding caffemodel file
            for secondary_file in self.model_files.secondary_files:
                if secondary_file.endswith(".caffemodel"):
                    caffemodel_path = secondary_file
                    break

            if not caffemodel_path:
                raise ValueError("Caffe model missing .caffemodel weight file")

            print(
                f"Loading Caffe model: prototxt={prototxt_path}, caffemodel={caffemodel_path}"
            )
            self.rknn.load_caffe(prototxt_path, blobs=caffemodel_path)

        elif self.current_model_type == ModelType.PYTORCH:
            print(">>>>>>>>>", self.rknn_config.torch_config())
            self.rknn.load_pytorch(self.model_path, **(self.rknn_config.torch_config()))
        elif self.current_model_type == ModelType.TENSORFLOW:
            # Step 1: Convert TensorFlow model to TFLite
            tflite_model = self._save_model2tflite(self.model_path)
            self.rknn.load_tflite(
                tflite_model,  # Input size
            )
        elif self.current_model_type == ModelType.DARKNET:
            # Darknet model requires cfg and weights files
            cfg_path = self.model_path  # Primary file is cfg
            weights_path = None

            # Find corresponding weights file
            for secondary_file in self.model_files.secondary_files:
                if secondary_file.endswith(".weights"):
                    weights_path = secondary_file
                    break

            if not weights_path:
                raise ValueError("Darknet model missing .weights weight file")

            print(f"Loading Darknet model: cfg={cfg_path}, weights={weights_path}")
            self.rknn.load_darknet(cfg=cfg_path, weight=weights_path)

        else:
            raise ValueError(f"Unsupported model type: {self.current_model_type}")


if __name__ == "__main__":
    from utils.config import ModelFiles

    configs = RKNNConverterConfig()
    # Example: single file model
    model_files = ModelFiles(
        primary_file="/home/dq/github/PaddleOCR/inference/rec_onnx/rec_fc_sim.onnx",
        secondary_files=[],
        model_type="onnx",
    )
    converter = RKNNConverter(
        model_files=model_files, output_path="./models.rknn", config=configs
    )
    converter.convert()
