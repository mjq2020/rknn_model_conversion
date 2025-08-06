#!/usr/bin/env python3
"""
Model file analyzer
Used to identify and organize multi-file models
"""

import os
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from utils.config import ModelFiles


class ModelType:
    """Model type constants"""

    ONNX = "onnx"
    TFLITE = "tflite"
    CAFFE = "caffe"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    DARKNET = "darknet"
    UNKNOWN = "unknown"


@dataclass
class FileInfo:
    """File information"""

    filename: str
    filepath: str
    extension: str
    size: int


class ModelFileAnalyzer:
    """Model file analyzer"""

    # File association rules for multi-file model formats
    MULTI_FILE_PATTERNS = {
        ModelType.CAFFE: {
            "primary": [".prototxt"],
            "secondary": [".caffemodel"],
            "pairs": [(".prototxt", ".caffemodel")],
        },
        ModelType.TENSORFLOW: {
            "primary": [".pb", ".meta"],
            "secondary": [".ckpt", ".data", ".index"],
            "groups": ["checkpoint", ".meta", ".data", ".index"],
        },
        # ModelType.TENSORFLOW: {
        #     'primary': ['.pb', '.meta'],
        #     'secondary': ['.ckpt', '.data', '.index'],
        #     'groups': ['checkpoint', '.meta', '.data', '.index']
        # },
        ModelType.DARKNET: {
            "primary": [".cfg"],
            "secondary": [".weights"],
            "pairs": [(".cfg", ".weights")],
        },
    }

    # Single-file model formats
    SINGLE_FILE_PATTERNS = {
        ModelType.ONNX: [".onnx"],
        ModelType.TFLITE: [".tflite"],
        ModelType.PYTORCH: [".pt", ".pth", ".pytorch"],
    }

    def __init__(self):
        pass

    def analyze_uploaded_files(self, uploaded_files: List[Dict]) -> List[ModelFiles]:
        """
        Analyze uploaded files and organize them into model file groups

        Args:
            uploaded_files: List of uploaded files, each containing filename, filepath, size, etc.

        Returns:
            List[ModelFiles]: List of organized model file groups
        """
        # Convert to FileInfo objects
        file_infos = []
        for file_data in uploaded_files:
            file_info = FileInfo(
                filename=file_data["original_name"],
                filepath=file_data["path"],
                extension=os.path.splitext(file_data["original_name"])[1].lower(),
                size=file_data["size"],
            )
            file_infos.append(file_info)

        # Analyze file groups
        model_groups = self._group_model_files(file_infos)

        return model_groups

    def _group_model_files(self, file_infos: List[FileInfo]) -> List[ModelFiles]:
        """Organize file information into model file groups"""
        model_groups = []
        used_files = set()
        if len(file_infos) == 3:
            pb_file = None
            index_file = None
            data_file = None
            for file_info in file_infos:
                if file_info.extension == ".pb":
                    pb_file = file_info.filepath
                elif file_info.extension == ".index":
                    index_file = file_info.filepath
                elif file_info.extension == ".data-00000-of-00001":
                    data_file = file_info.filepath
            if pb_file and index_file and data_file:
                model_groups.append(
                    ModelFiles(
                        primary_file=pb_file,
                        secondary_files=[data_file],
                        additional_files=[index_file],
                        model_type=ModelType.TENSORFLOW,
                    )
                )
            return model_groups

        # 1. Process multi-file models
        for model_type, patterns in self.MULTI_FILE_PATTERNS.items():
            groups = self._find_multi_file_groups(
                file_infos, model_type, patterns, used_files
            )
            model_groups.extend(groups)

        # 2. Process remaining single-file models
        for file_info in file_infos:
            if file_info.filepath not in used_files:
                model_type = self._identify_single_file_type(file_info.extension)
                if model_type != ModelType.UNKNOWN:
                    model_group = ModelFiles(
                        primary_file=file_info.filepath,
                        secondary_files=[],
                        model_type=model_type,
                    )
                    model_groups.append(model_group)
                    used_files.add(file_info.filepath)

        return model_groups

    def _find_multi_file_groups(
        self,
        file_infos: List[FileInfo],
        model_type: str,
        patterns: Dict,
        used_files: set,
    ) -> List[ModelFiles]:
        """Find multi-file model groups"""
        groups = []

        if "pairs" in patterns:
            # Process paired files (e.g., Caffe's .prototxt and .caffemodel)
            groups.extend(
                self._find_paired_files(
                    file_infos, model_type, patterns["pairs"], used_files
                )
            )

        if "groups" in patterns:
            # Process file groups (e.g., TensorFlow's checkpoint series files)
            groups.extend(
                self._find_grouped_files(
                    file_infos, model_type, patterns["groups"], used_files
                )
            )

        return groups

    def _find_paired_files(
        self,
        file_infos: List[FileInfo],
        model_type: str,
        pairs: List[Tuple[str, str]],
        used_files: set,
    ) -> List[ModelFiles]:
        """Find paired files"""
        groups = []

        for primary_ext, secondary_ext in pairs:
            # Find primary files
            primary_files = [
                f
                for f in file_infos
                if f.extension == primary_ext and f.filepath not in used_files
            ]

            for primary_file in primary_files:
                # Find corresponding auxiliary files
                secondary_files = []

                # First try exact base name matching
                base_name = os.path.splitext(primary_file.filename)[0]
                for file_info in file_infos:
                    if (
                        file_info.extension == secondary_ext
                        and file_info.filepath not in used_files
                        and os.path.splitext(file_info.filename)[0] == base_name
                    ):
                        secondary_files.append(file_info.filepath)

                # If no exact match found, try fuzzy matching
                if not secondary_files:
                    secondary_files = self._find_fuzzy_match(
                        primary_file, secondary_ext, file_infos, used_files
                    )

                if secondary_files:
                    # Found complete file pair
                    model_group = ModelFiles(
                        primary_file=primary_file.filepath,
                        secondary_files=secondary_files,
                        model_type=model_type,
                    )
                    groups.append(model_group)

                    # Mark as used
                    used_files.add(primary_file.filepath)
                    used_files.update(secondary_files)

        return groups

    def _find_fuzzy_match(
        self,
        primary_file: FileInfo,
        secondary_ext: str,
        file_infos: List[FileInfo],
        used_files: set,
    ) -> List[str]:
        """Fuzzy match auxiliary files"""
        secondary_files = []
        primary_base = os.path.splitext(primary_file.filename)[0].lower()

        # For Caffe models, try multiple matching strategies
        for file_info in file_infos:
            if (
                file_info.extension == secondary_ext
                and file_info.filepath not in used_files
            ):

                secondary_base = os.path.splitext(file_info.filename)[0].lower()

                # Strategy 1: Check for common keywords
                if self._has_common_keywords(primary_base, secondary_base):
                    secondary_files.append(file_info.filepath)
                    break

                # Strategy 2: Check if one name contains the other
                elif primary_base in secondary_base or secondary_base in primary_base:
                    secondary_files.append(file_info.filepath)
                    break

        # If still not found and there's only one file with the corresponding extension, pair it
        if not secondary_files:
            matching_files = [
                f
                for f in file_infos
                if f.extension == secondary_ext and f.filepath not in used_files
            ]
            if len(matching_files) == 1:
                secondary_files.append(matching_files[0].filepath)

        return secondary_files

    def _has_common_keywords(self, name1: str, name2: str) -> bool:
        """Check if two filenames have common keywords"""
        # Common model name keywords
        keywords = [
            "googlenet",
            "resnet",
            "vgg",
            "alexnet",
            "mobilenet",
            "squeezenet",
            "densenet",
            "inception",
            "yolo",
            "ssd",
        ]

        for keyword in keywords:
            if keyword in name1 and keyword in name2:
                return True

        # Check for common substrings of 3+ characters
        for i in range(len(name1) - 2):
            for j in range(
                3, min(len(name1) - i + 1, 8)
            ):  # Check up to 7 characters max
                substr = name1[i : i + j]
                if len(substr) >= 3 and substr in name2:
                    return True

        return False

    def _find_grouped_files(
        self,
        file_infos: List[FileInfo],
        model_type: str,
        group_patterns: List[str],
        used_files: set,
    ) -> List[ModelFiles]:
        """Find file groups (e.g., TensorFlow checkpoint series)"""
        groups = []

        # Group by base name
        base_groups = {}
        for file_info in file_infos:
            if file_info.filepath in used_files:
                continue

            # Check if it matches any group pattern
            matches_pattern = False
            for pattern in group_patterns:
                if pattern in file_info.filename or file_info.extension in pattern:
                    matches_pattern = True
                    break

            if matches_pattern:
                # Extract base name
                base_name = self._extract_base_name(file_info.filename, group_patterns)
                if base_name not in base_groups:
                    base_groups[base_name] = []
                base_groups[base_name].append(file_info)

        # Create model group for each base name
        for base_name, files in base_groups.items():
            if len(files) > 1:  # Only form groups with multiple files
                # Select primary file (usually .meta or .pb file)
                primary_file = None
                secondary_files = []

                for file_info in files:
                    if file_info.extension in [".meta", ".pb"]:
                        primary_file = file_info.filepath
                    else:
                        secondary_files.append(file_info.filepath)

                if primary_file:
                    model_group = ModelFiles(
                        primary_file=primary_file,
                        secondary_files=secondary_files,
                        model_type=model_type,
                    )
                    groups.append(model_group)

                    # Mark as used
                    for file_info in files:
                        used_files.add(file_info.filepath)

        return groups

    def _extract_base_name(self, filename: str, patterns: List[str]) -> str:
        """Extract base name from filename"""
        base_name = filename

        # Remove extension
        base_name = os.path.splitext(base_name)[0]

        # Remove common suffixes
        suffixes_to_remove = ["-00000-of-00001", ".data", ".index", ".meta"]
        for suffix in suffixes_to_remove:
            if base_name.endswith(suffix):
                base_name = base_name[: -len(suffix)]

        return base_name

    def _identify_single_file_type(self, extension: str) -> str:
        """Identify single-file model type"""
        for model_type, extensions in self.SINGLE_FILE_PATTERNS.items():
            if extension in extensions:
                return model_type
        return ModelType.UNKNOWN

    def validate_model_files(self, model_files: ModelFiles) -> Tuple[bool, str]:
        """
        Validate the integrity of model file groups

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        # Check if primary file exists
        if not os.path.exists(model_files.primary_file):
            return False, f"Primary file does not exist: {model_files.primary_file}"

        # Check if auxiliary files exist
        for secondary_file in model_files.secondary_files:
            if not os.path.exists(secondary_file):
                return False, f"Auxiliary file does not exist: {secondary_file}"

        # Perform model type-specific validation
        if model_files.model_type == ModelType.CAFFE:
            return self._validate_caffe_files(model_files)
        elif model_files.model_type == ModelType.DARKNET:
            return self._validate_darknet_files(model_files)
        elif model_files.model_type == ModelType.TENSORFLOW:
            return self._validate_tensorflow_files(model_files)

        return True, ""

    def _validate_caffe_files(self, model_files: ModelFiles) -> Tuple[bool, str]:
        """Validate Caffe model files"""
        primary_ext = os.path.splitext(model_files.primary_file)[1].lower()

        if primary_ext == ".prototxt":
            # Primary file is prototxt, should have corresponding caffemodel
            if not model_files.secondary_files:
                return False, "Caffe model missing .caffemodel weight file"

            caffemodel_found = any(
                f.endswith(".caffemodel") for f in model_files.secondary_files
            )
            if not caffemodel_found:
                return False, "Caffe model missing .caffemodel weight file"

        return True, ""

    def _validate_darknet_files(self, model_files: ModelFiles) -> Tuple[bool, str]:
        """Validate Darknet model files"""
        primary_ext = os.path.splitext(model_files.primary_file)[1].lower()

        if primary_ext == ".cfg":
            # Primary file is cfg, should have corresponding weights
            if not model_files.secondary_files:
                return False, "Darknet model missing .weights weight file"

            weights_found = any(
                f.endswith(".weights") for f in model_files.secondary_files
            )
            if not weights_found:
                return False, "Darknet model missing .weights weight file"

        return True, ""

    def _validate_tensorflow_files(self, model_files: ModelFiles) -> Tuple[bool, str]:
        """Validate TensorFlow model files"""
        # TensorFlow model validation is relatively complex, basic checks here
        all_files = model_files.get_all_files()
        extensions = [os.path.splitext(f)[1].lower() for f in all_files]

        # Check for necessary file types
        has_graph = any(ext in [".pb", ".meta"] for ext in extensions)
        if not has_graph:
            return (
                False,
                "TensorFlow model missing graph definition file (.pb or .meta)",
            )

        return True, ""


# Global analyzer instance
model_analyzer = ModelFileAnalyzer()
