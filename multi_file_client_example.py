#!/usr/bin/env python3
"""
Multi-file Model Upload Client Example
Demonstrates how to upload multi-file models like Caffe, Darknet, etc.
"""

import requests
import json
import os
from typing import List, Dict


def upload_caffe_model(prototxt_path: str, caffemodel_path: str, config: Dict = None):
    """Upload Caffe model (prototxt + caffemodel)"""
    url = "http://127.0.0.1:8080/api/upload_and_create_task"

    if not os.path.exists(prototxt_path):
        print(f"❌ prototxt file does not exist: {prototxt_path}")
        return None

    if not os.path.exists(caffemodel_path):
        print(f"❌ caffemodel file does not exist: {caffemodel_path}")
        return None

    # Default configuration
    if config is None:
        config = {
            "target_platform": "rk3588",
            "quantized_dtype": "w8a8",
            "do_quantization": True,
            "dataset": "./images.txt",
        }

    data = {"config": json.dumps(config)}

    try:
        with open(prototxt_path, "rb") as prototxt_file, open(
            caffemodel_path, "rb"
        ) as caffemodel_file:

            files = [
                (
                    "file",
                    (
                        os.path.basename(prototxt_path),
                        prototxt_file,
                        "application/octet-stream",
                    ),
                ),
                (
                    "file",
                    (
                        os.path.basename(caffemodel_path),
                        caffemodel_file,
                        "application/octet-stream",
                    ),
                ),
            ]
            files = {"file1": prototxt_file, "file2": caffemodel_file}

            print(f"🚀 Uploading Caffe model:")
            print(f"   prototxt: {os.path.basename(prototxt_path)}")
            print(f"   caffemodel: {os.path.basename(caffemodel_path)}")

            response = requests.post(url, data=data, files=files)

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Task created successfully!")
            print(f"   Task ID: {result['task_id']}")
            print(f"   Model type: {result['model_type']}")
            print(f"   Primary file: {result['files_info']['primary_file']}")
            print(f"   Secondary files: {result['files_info']['secondary_files']}")
            print(f"   Output path: {result['output_path']}")
            return result["task_id"]
        else:
            print(f"❌ Failed to create task: {response.json()}")
            return None
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return None


def upload_darknet_model(cfg_path: str, weights_path: str, config: Dict = None):
    """Upload Darknet model (cfg + weights)"""
    url = "http://127.0.0.1:8080/api/upload_and_create_task"

    if not os.path.exists(cfg_path):
        print(f"❌ cfg file does not exist: {cfg_path}")
        return None

    if not os.path.exists(weights_path):
        print(f"❌ weights file does not exist: {weights_path}")
        return None

    # Default configuration
    if config is None:
        config = {
            "target_platform": "rk3588",
            "quantized_dtype": "w8a8",
            "do_quantization": True,
            "dataset": "./images.txt",
        }

    data = {"config": json.dumps(config)}

    try:
        with open(cfg_path, "rb") as cfg_file, open(weights_path, "rb") as weights_file:

            files = [
                (
                    "file",
                    (os.path.basename(cfg_path), cfg_file, "application/octet-stream"),
                ),
                (
                    "file",
                    (
                        os.path.basename(weights_path),
                        weights_file,
                        "application/octet-stream",
                    ),
                ),
            ]

            print(f"🚀 Uploading Darknet model:")
            print(f"   cfg: {os.path.basename(cfg_path)}")
            print(f"   weights: {os.path.basename(weights_path)}")

            response = requests.post(url, data=data, files=files)

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Task created successfully!")
            print(f"   Task ID: {result['task_id']}")
            print(f"   Model type: {result['model_type']}")
            print(f"   Primary file: {result['files_info']['primary_file']}")
            print(f"   Secondary files: {result['files_info']['secondary_files']}")
            print(f"   Output path: {result['output_path']}")
            return result["task_id"]
        else:
            print(f"❌ Failed to create task: {response.json()}")
            return None
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return None


def upload_single_file_model(model_path: str, config: Dict = None):
    """Upload single-file model (ONNX, TFLite, PyTorch, etc.)"""
    url = "http://127.0.0.1:8080/api/upload_and_create_task"

    if not os.path.exists(model_path):
        print(f"❌ Model file does not exist: {model_path}")
        return None

    # Default configuration
    if config is None:
        config = {
            "target_platform": "rk3588",
            "quantized_dtype": "w8a8",
            "do_quantization": True,
            "dataset": "./images.txt",
        }

    data = {"config": json.dumps(config)}

    try:
        with open(model_path, "rb") as model_file:
            files = {
                "file": (
                    os.path.basename(model_path),
                    model_file,
                    "application/octet-stream",
                )
            }

            print(f"🚀 Uploading single-file model:")
            print(f"   File: {os.path.basename(model_path)}")

            response = requests.post(url, data=data, files=files)

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Task created successfully!")
            print(f"   Task ID: {result['task_id']}")
            print(f"   Model type: {result['model_type']}")
            print(f"   Primary file: {result['files_info']['primary_file']}")
            print(f"   Output path: {result['output_path']}")
            return result["task_id"]
        else:
            print(f"❌ Failed to create task: {response.json()}")
            return None
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return None


def upload_tensorflow_model(
    pb_path: str, additional_files: List[str] = None, config: Dict = None
):
    """Upload TensorFlow model (may include multiple files)"""
    url = "http://127.0.0.1:8080/api/upload_and_create_task"

    if not os.path.exists(pb_path):
        print(f"❌ pb file does not exist: {pb_path}")
        return None

    # Default configuration
    if config is None:
        config = {
            "target_platform": "rk3588",
            "quantized_dtype": "w8a8",
            "do_quantization": True,
            "dataset": "./images.txt",
        }

    data = {"config": json.dumps(config)}

    try:
        files_to_upload = []

        # Add primary file
        with open(pb_path, "rb") as pb_file:
            files_to_upload.append(
                (
                    "file",
                    (
                        os.path.basename(pb_path),
                        pb_file.read(),
                        "application/octet-stream",
                    ),
                )
            )

        # Add additional files
        if additional_files:
            for file_path in additional_files:
                if os.path.exists(file_path):
                    with open(file_path, "rb") as additional_file:
                        files_to_upload.append(
                            (
                                "file",
                                (
                                    os.path.basename(file_path),
                                    additional_file.read(),
                                    "application/octet-stream",
                                ),
                            )
                        )

        print(f"🚀 Uploading TensorFlow model:")
        print(f"   Primary file: {os.path.basename(pb_path)}")
        print(">>>>>", additional_files)
        if additional_files:
            print(
                f"   Additional files: {[os.path.basename(f) for f in additional_files if os.path.exists(f)]}"
            )
        print(len(files_to_upload))
        response = requests.post(url, data=data, files=files_to_upload)

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Task created successfully!")
            print(f"   Task ID: {result['task_id']}")
            print(f"   Model type: {result['model_type']}")
            print(f"   Primary file: {result['files_info']['primary_file']}")
            print(f"   Secondary files: {result['files_info']['secondary_files']}")
            print(f"   Output path: {result['output_path']}")
            return result["task_id"]
        else:
            print(f"❌ Failed to create task: {response.json()}")
            return None
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return None


def wait_for_completion(task_id: str):
    """Wait for task completion"""
    import time

    url = f"http://127.0.0.1:8080/api/tasks/{task_id}"

    print(f"⏳ Waiting for task {task_id} to complete...")

    while True:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                task_info = response.json()
                status = task_info["status"]
                progress = task_info.get("progress", 0)

                print(f"   Status: {status}, Progress: {progress:.1f}%")

                if status == "completed":
                    print(f"✅ Task completed!")
                    print(f"   Result file: {task_info.get('result_path', 'N/A')}")
                    return True
                elif status == "failed":
                    print(
                        f"❌ Task failed: {task_info.get('error_message', 'Unknown error')}"
                    )
                    return False
                elif status == "cancelled":
                    print(f"⚠️ Task cancelled")
                    return False
            else:
                print(f"❌ Failed to query task status: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Task status query exception: {e}")
            return False

        time.sleep(5)


def main():
    """Main function - demonstrates multi-file model upload"""
    print("🚀 Multi-file Model Upload Example")
    print("=" * 60)

    # Example file paths (please modify according to actual situation)
    examples = {
        "caffe": {
            "prototxt": "/home/dq/github/ModelAssistant/work_dirs/googlenet.prototxt",
            "caffemodel": "/home/dq/github/ModelAssistant/work_dirs/bvlc_googlenet.caffemodel",
        },
        "darknet": {"cfg": "/path/to/model.cfg", "weights": "/path/to/model.weights"},
        "onnx": "/path/to/model.onnx",
        "tflite": "/path/to/model.tflite",
        "tensorflow": {
            "pb": r"/home/dq/github/ModelAssistant/work_dirs/saved_model.pb",
            "index": r"/home/dq/github/ModelAssistant/work_dirs/variables.index",
            "data": r"/home/dq/github/ModelAssistant/work_dirs/variables.data-00000-of-00001",
        },
    }

    print("\n📝 Available examples:")
    print("1. Caffe model (prototxt + caffemodel)")
    print("2. Darknet model (cfg + weights)")
    print("3. ONNX model (single file)")
    print("4. TFLite model (single file)")

    # Test with actual file paths here
    print(
        "\n⚠️ Please modify the file paths in the example to actual paths before running tests"
    )

    # Example: If you have actual file paths, you can call like this
    # task_id = upload_caffe_model(
    #     prototxt_path=examples["caffe"]["prototxt"],
    #     caffemodel_path=examples["caffe"]["caffemodel"],
    # )
    task_id = upload_tensorflow_model(
        pb_path=examples["tensorflow"]["pb"],
        additional_files=[
            examples["tensorflow"]["index"],
            examples["tensorflow"]["data"],
        ],
    )

    if task_id:
        wait_for_completion(task_id)


if __name__ == "__main__":
    main()
