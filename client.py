import requests
import time
import json


def upload_model():
    url = "http://127.0.0.1:8080/api/upload"
    model_path = r"/home/dq/github/PaddleOCR/inference/rec_onnx/rec_fc_sim_new.onnx"

    try:
        with open(model_path, "rb") as f:
            files = {"file": f}
            response = requests.post(url, files=files)

        print(response.json())
    except Exception as e:
        print(f"Upload failed: {e}")


def download_result(task_id):
    url = f"http://127.0.0.1:8080/api/download/{task_id}"
    response = requests.get(url)
    if response.status_code == 200:
        with open(f"./models_{task_id}.rknn", "wb") as f:
            f.write(response.content)
    else:
        print(response.json())


def query_task():
    url = "http://127.0.0.1:8080/api/tasks"
    response = requests.get(url)

    if response.status_code == 200:
        result = response.json()
        tasks = result.get("tasks", [])

        print(f"Query task list: {len(tasks)} tasks in total")
        print("-" * 60)

        current_tasks = []
        historical_tasks = []

        for task in tasks:
            if task.get("is_historical", False):
                historical_tasks.append(task)
            else:
                current_tasks.append(task)

        # Display current tasks
        if current_tasks:
            print(f"📋 Current tasks ({len(current_tasks)} tasks):")
            for task in current_tasks:
                print(
                    f"  🆔 {task['task_id']} - {task['status']} - {task.get('progress', 0)}%"
                )

        # Display historical tasks
        if historical_tasks:
            print(f"\n📚 Historical tasks ({len(historical_tasks)} tasks):")
            for task in historical_tasks:
                model_name = task.get("model_name", "N/A")
                completed_at = task.get("completed_at", "N/A")
                print(
                    f"  🆔 {task['task_id']} - {model_name} - Completed at: {completed_at}"
                )

        print("-" * 60)
    else:
        print("Failed to query task list:", response.json())


def query_task_by_id(task_id):
    url = f"http://127.0.0.1:8080/api/tasks/{task_id}"
    response = requests.get(url)

    if response.status_code == 200:
        task = response.json()

        print(f"Query task details: {task_id}")
        print("-" * 40)
        print(f"Status: {task['status']}")
        print(f"Progress: {task.get('progress', 0)}%")

        if task.get("is_historical", False):
            print(f"Type: Historical task")
            print(f"Model name: {task.get('model_name', 'N/A')}")
            print(f"Completion time: {task.get('completed_at', 'N/A')}")
        else:
            print(f"Type: Current task")
            print(f"Creation time: {task.get('created_at', 'N/A')}")
            print(f"Start time: {task.get('started_at', 'N/A')}")
            print(f"Completion time: {task.get('completed_at', 'N/A')}")

        if task.get("error_message"):
            print(f"Error message: {task['error_message']}")

        if task.get("result_path"):
            print(f"Result path: {task['result_path']}")

        print("-" * 40)
        return task
    else:
        print("Failed to query task details:", response.json())
        return None


def create_task():
    url = "http://127.0.0.1:8080/api/tasks"

    # Optional conversion configuration
    config = {
        "target_platform": "rk3588",
        "quantized_dtype": "w8a8",
        "do_quantization": True,
        "dataset": "./images.txt",
    }

    data = {
        "model_path": r"/home/dq/github/ModelAssistant/work_dirs/googlenet.prototxt",
        "config": config,  # Optional: use default configuration if not provided
        # No need to specify output_path anymore, server will auto-generate
    }
    response = requests.post(url, json=data)
    print("Create task:", response.json())
    if response.status_code == 200:
        result = response.json()
        task_id = result["task_id"]
        print(f"Expected output path: {result.get('output_path', 'N/A')}")
        return task_id
    else:
        return None


def get_task_logs(task_id):
    url = f"http://127.0.0.1:8080/api/tasks/{task_id}/logs"
    response = requests.get(url)
    print("Query task logs:", response.json())


def upload_and_create_task():
    url = "http://127.0.0.1:8080/api/upload_and_create_task"
    model_path = r"/home/dq/github/PaddleOCR/inference/rec_onnx/rec_fc_sim_new.onnx"

    # Prepare form data (optional configuration)
    config = {
        "target_platform": "rv1106",
        "quantized_dtype": "w8a8",
        "do_quantization": True,
        "dataset": "./images.txt",  # Can specify dataset path in config
        "input_size_list": [[1, 3, 640, 640]],
    }

    data = {
        "config": json.dumps(
            config
        )  # Optional: conversion configuration, use default if not provided
        # No need to specify output_path anymore, server will auto-generate
    }

    try:
        with open(model_path, "rb") as f:
            files = {"file": f}

            # Send request using multipart/form-data format
            response = requests.post(url, data=data, files=files)

        print("Upload and create task:", response.json())
        print()
        if response.status_code == 200:
            result = response.json()
            task_id = result["task_id"]
            print(f"Expected output path: {result.get('output_path', 'N/A')}")
            return task_id
        else:
            return None
    except Exception as e:
        print(f"Upload failed: {e}")
        return None


if __name__ == "__main__":
    # upload_model()
    # # download_result()
    task_id = upload_and_create_task()
    query_task()
    while True:
        task_info = query_task_by_id(task_id)
        if task_info["status"] == "completed":
            break
        time.sleep(5)
    download_result(task_id)

    get_task_logs(task_id)
