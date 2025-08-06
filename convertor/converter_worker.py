import asyncio
import os
import shutil
import os.path as osp
from typing import Tuple, Optional
from datetime import datetime
import tempfile

from task_manager import TaskInfo
from utils.config import ConversionTask, ModelFiles
from utils.logger import TaskLogger
from convertor.converter import RKNNConverter


class ConverterWorker:
    """Conversion worker"""

    def __init__(self, task_info: TaskInfo, task_logger: TaskLogger):
        self.task_info = task_info
        self.task = task_info.task
        self.logger = task_logger
        self.progress = 0.0

    async def convert(self) -> Tuple[bool, Optional[str]]:
        """Execute model conversion"""
        try:
            self.logger.info("Starting model conversion process")

            # Validate input files
            if not await self._validate_input():
                return False, "Input file validation failed"

            # Prepare output path
            output_path = await self._prepare_output_path()

            # Execute conversion
            success, error = await self._perform_conversion(output_path)

            if success:
                self.logger.info(f"Conversion successful, output file: {output_path}")
                return True, output_path
            else:
                return False, error

        except Exception as e:
            error_msg = f"Exception occurred during conversion: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

    async def _validate_input(self) -> bool:
        """Validate input files"""
        self.logger.info("Validating input files...")

        # Check if model file exists
        if not os.path.exists(self.task.model_path):
            self.logger.error(f"Model file does not exist: {self.task.model_path}")
            return False

        # Check file size
        file_size = os.path.getsize(self.task.model_path)
        if file_size == 0:
            self.logger.error("Model file is empty")
            return False

        # Check file extension
        _, ext = os.path.splitext(self.task.model_path)
        supported_extensions = {
            ".onnx",
            ".tflite",
            ".prototxt",
            ".pytorch",
            ".pb",
            ".pt",
            ".pth",
            ".darknet",
        }

        if ext.lower() not in supported_extensions:
            self.logger.error(f"Unsupported model format: {ext}")
            return False

        self.logger.info(
            f"Input file validation passed, file size: {file_size / 1024 / 1024:.2f}MB"
        )
        return True

    async def _prepare_output_path(self) -> str:
        """Prepare output path"""
        self.logger.info("Preparing output path...")

        # Use task's get_output_path method to generate output path
        from utils.config import DEFAULT_SERVER_CONFIG

        output_path = self.task.get_output_path(DEFAULT_SERVER_CONFIG.output_folder)

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        self.logger.info(f"Output path: {output_path}")
        return output_path

    async def _perform_conversion(self, output_path: str) -> Tuple[bool, Optional[str]]:
        """Execute actual conversion work"""
        self.logger.info("Starting model conversion execution...")

        try:
            # Execute conversion in thread pool (avoid blocking event loop)
            loop = asyncio.get_event_loop()
            success, error = await loop.run_in_executor(
                None, self._convert_sync, output_path
            )

            return success, error

        except Exception as e:
            error_msg = f"Conversion execution failed: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

    def _convert_sync(self, output_path: str) -> Tuple[bool, Optional[str]]:
        """Execute conversion synchronously (run in thread pool)"""
        try:
            self.logger.info("Loading model...")
            self._update_progress(10)
            if self.task.model_files.additional_files:
                with tempfile.TemporaryDirectory() as temp_dir:
                    os.makedirs(osp.join(temp_dir, "variables"), exist_ok=True)
                    temp_index = osp.join(temp_dir, "variables", "variables.index")
                    temp_data = osp.join(
                        temp_dir, "variables", "variables.data-00000-of-00001"
                    )
                    temp_pb = osp.join(temp_dir, "saved_model.pb")
                    shutil.copy(self.task.model_files.secondary_files[0], temp_data)
                    shutil.copy(self.task.model_files.additional_files[0], temp_index)
                    shutil.copy(self.task.model_files.primary_file, temp_pb)
                    # Create converter
                    converter = RKNNConverter(
                        model_files=ModelFiles(
                            primary_file=temp_pb,
                            secondary_files=[temp_data],
                            additional_files=[temp_index],
                        ),
                        output_path=output_path,
                        dataset_path=self.task.get_dataset_path(),
                        config=self.task.config,
                    )

                    self.logger.info("Starting conversion...")
                    self._update_progress(30)

                    # Execute conversion
                    success, error = converter.convert(self._update_progress)

                    if success:
                        self.logger.info("Conversion completed")
                        return True, None
                    else:
                        self.logger.error(f"Conversion failed: {error}")
                        return False, str(error)

            # Create converter
            converter = RKNNConverter(
                model_files=self.task.model_files,
                output_path=output_path,
                dataset_path=self.task.get_dataset_path(),
                config=self.task.config,
            )

            self.logger.info("Starting conversion...")
            self._update_progress(30)

            # Execute conversion
            success, error = converter.convert(self._update_progress)

            if success:
                self.logger.info("Conversion completed")
                return True, None
            else:
                self.logger.error(f"Conversion failed: {error}")
                return False, str(error)

        except Exception as e:
            error_msg = f"Conversion process exception: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

    def _update_progress(self, progress: float):
        """Update progress"""
        self.progress = progress
        self.logger.info(f"Conversion progress: {progress}%")
        self.task_info.progress = progress

        # Progress callback notification can be added here
        # For example: self.task_manager.update_task_progress(self.task.task_id, progress)
