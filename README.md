# RKNN Model Conversion Daemon

An elegant, modular RKNN model conversion daemon that provides network interface services and supports conversion of various model formats.

## ✨ Core Features

- 🚀 **Asynchronous Processing Architecture**: High-performance asynchronous architecture based on asyncio
- 🔄 **Multi-task Concurrency**: Support for concurrent processing of multiple conversion tasks
- 📊 **Real-time Monitoring**: Provides real-time task status and progress monitoring
- 📁 **Intelligent File Management**: Support for model file upload and result download
- 🧠 **Automatic Model Analysis**: Intelligent recognition and processing of multi-file model formats
- 📝 **Detailed Logging System**: Complete task logging system
- 🛡️ **Error Handling Mechanism**: Comprehensive error handling and recovery mechanisms
- 🔧 **Flexible Configuration Options**: Support for various conversion configuration options

## 📋 Supported Model Formats

### Single-file Models
- **ONNX** (`.onnx`) - Open Neural Network Exchange format
- **TensorFlow Lite** (`.tflite`) - Lightweight TensorFlow models
- **PyTorch** (`.pt`, `.pth`, `.pytorch`) - PyTorch model files

### Multi-file Models
- **Caffe** (`.prototxt` + `.caffemodel`) - Network structure file + weight file
- **Darknet** (`.cfg` + `.weights`) - Configuration file + weight file
- **TensorFlow** (`.pb` + related files) - Graph definition file + weight files
  - Support for Frozen Graph (`.pb`)
  - Support for SavedModel format
  - Support for Checkpoint format (`.meta` + `.ckpt` + `.index` + `.data`)

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Server    │    │  Task Manager   │    │ Converter Worker│
│                 │    │                 │    │                 │
│ - HTTP Interface│◄──►│ - Task Queue Mgmt│◄──►│ - Model Convert │
│ - File Up/Down  │    │ - Status Tracking│    │ - Progress Update│
│ - Multi-file    │    │ - Worker Pool   │    │ - Error Handling│
│   Support       │    │ - History Mgmt  │    │ - RKNN Core     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Logger        │    │   Config        │    │ Model Analyzer  │
│                 │    │                 │    │                 │
│ - Unified Log   │    │ - Config Mgmt   │    │ - Format Detect │
│ - Task Log      │    │ - Param Valid   │    │ - File Grouping │
│ - Color Output  │    │ - Default Config│    │ - Validation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📦 Installation

### System Requirements
- Python 3.7+
- RKNN Toolkit2 1.4.0+

### Installation Steps

```bash
# Clone the project
git clone <repository-url>
cd rknn_model_conversion

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p uploads outputs temp logs
```

### Using Startup Script (Recommended)

```bash
# Use the provided startup script to automatically check environment and dependencies
chmod +x start_server.sh
./start_server.sh
```

## 🚀 Quick Start

### 1. Start Server

```bash
# Start with default configuration
python main.py

# Start with custom configuration
python main.py --host 0.0.0.0 --port 8080 --workers 4

# Enable debug mode
python main.py --debug
```


## 📚 API Documentation

### Health Check
```http
GET /health
```

Response:
```json
{
    "status": "healthy",
    "timestamp": "2024-01-01T12:00:00",
    "version": "1.0.0"
}
```

### Task Management

#### Create Conversion Task (Local File)
```http
POST /api/tasks
Content-Type: application/json

{
    "model_path": "/path/to/model.onnx",
    "config": {
        "target_platform": "rk3588",
        "do_quantization": true,
        "dataset": "./images.txt",
        "quantized_dtype": "w8a8"
    },
    "callback_url": "http://example.com/callback"
}
```

#### Upload and Create Task (Recommended)
```http
POST /api/upload_and_create_task
Content-Type: multipart/form-data

file: [model file(s)]
config: {
    "target_platform": "rk3588",
    "do_quantization": true,
    "dataset": "./images.txt"
}
```

#### Get Task List
```http
GET /api/tasks
```

#### Get Task Details
```http
GET /api/tasks/{task_id}
```

#### Cancel Task
```http
DELETE /api/tasks/{task_id}
```

### File Management

#### Upload Model File
```http
POST /api/upload
Content-Type: multipart/form-data

file: [model file]
```

#### Download Conversion Result
```http
GET /api/download/{task_id}
```

#### Get Task Logs
```http
GET /api/tasks/{task_id}/logs
```

## 💻 Usage Examples

### Python Client Examples

#### Single-file Model Conversion
```python
import requests
import json
import time

def convert_onnx_model(model_path):
    url = "http://127.0.0.1:8080/api/upload_and_create_task"
    
    config = {
        "target_platform": "rk3588",
        "quantized_dtype": "w8a8",
        "do_quantization": True,
        "dataset": "./images.txt"
    }
    
    data = {"config": json.dumps(config)}
    
    with open(model_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, data=data, files=files)
    
    if response.status_code == 200:
        result = response.json()
        task_id = result["task_id"]
        print(f"Task created successfully: {task_id}")
        return task_id
    else:
        print(f"Task creation failed: {response.json()}")
        return None

# Usage example
task_id = convert_onnx_model("model.onnx")
```

#### Multi-file Model Conversion (Caffe)
```python
def convert_caffe_model(prototxt_path, caffemodel_path):
    url = "http://127.0.0.1:8080/api/upload_and_create_task"
    
    config = {
        "target_platform": "rk3588",
        "quantized_dtype": "w8a8",
        "do_quantization": True
    }
    
    data = {"config": json.dumps(config)}
    
    with open(prototxt_path, "rb") as prototxt_file, \
         open(caffemodel_path, "rb") as caffemodel_file:
        
        files = {
            "file1": prototxt_file,
            "file2": caffemodel_file
        }
        
        response = requests.post(url, data=data, files=files)
    
    return response.json()

# Usage example
result = convert_caffe_model("model.prototxt", "model.caffemodel")
```

#### Task Status Monitoring
```python
def wait_for_completion(task_id):
    url = f"http://127.0.0.1:8080/api/tasks/{task_id}"
    
    while True:
        response = requests.get(url)
        if response.status_code == 200:
            task_info = response.json()
            status = task_info["status"]
            progress = task_info.get("progress", 0)
            
            print(f"Status: {status}, Progress: {progress:.1f}%")
            
            if status in ["completed", "failed", "cancelled"]:
                break
        
        time.sleep(5)
    
    return task_info

# Usage example
task_info = wait_for_completion(task_id)
```

### cURL Examples

```bash
# Health check
curl http://localhost:8080/health

# Upload single-file model
curl -X POST http://localhost:8080/api/upload_and_create_task \
  -F "file=@model.onnx" \
  -F 'config={"target_platform":"rk3588","do_quantization":true}'

# Upload multi-file model (Caffe)
curl -X POST http://localhost:8080/api/upload_and_create_task \
  -F "file=@model.prototxt" \
  -F "file=@model.caffemodel" \
  -F 'config={"target_platform":"rk3588","quantized_dtype":"w8a8"}'

# Query task status
curl http://localhost:8080/api/tasks/{task_id}

# Download result
curl -O http://localhost:8080/api/download/{task_id}
```

## ⚙️ Configuration Options

### Server Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| host | 0.0.0.0 | Server host address |
| port | 8080 | Server port |
| max_workers | 4 | Maximum worker threads |
| upload_folder | ./uploads | Upload file directory |
| output_folder | ./outputs | Output file directory |
| temp_folder | ./temp | Temporary file directory |
| max_file_size | 500MB | Maximum file size |

### Conversion Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| target_platform | rk3588 | Target platform |
| do_quantization | true | Whether to perform quantization |
| dataset | ./images.txt | Calibration dataset |
| mean_values | [0,0,0] | Mean values |
| std_values | [255,255,255] | Standard deviation |
| quantized_dtype | w8a8 | Quantization data type |

### Supported Target Platforms
- rk3588
- rk3568
- rk3566
- rv1106
- rv1103
- rk3562
- rk3576

## 📊 Logging System

The system provides two levels of logging:

1. **Global Log**: Records server runtime status and system events
2. **Task Log**: Each conversion task has an independent log file

Log file locations:
- Global log: `./logs/server.log`
- Task log: `./logs/task_{task_id}.log`

## 🛡️ Error Handling

The system provides comprehensive error handling mechanisms:

- Input file validation
- Conversion process exception capture
- Network request error handling
- Resource cleanup and recovery
- Automatic model format recognition and validation

## ⚡ Performance Optimization

- Asynchronous I/O processing
- Multi-threaded task execution
- File streaming transmission
- Memory usage optimization
- Intelligent task scheduling

## 🔒 Security Considerations

- File type validation
- File size limits
- Path security checks
- Error message filtering
- Upload file isolation

## 🔧 Troubleshooting

### Common Issues

1. **Port in use**
   ```bash
   # Check port usage
   netstat -tulpn | grep 8080
   
   # Start with different port
   python main.py --port 8081
   ```

2. **File permission issues**
   ```bash
   # Ensure directories have write permissions
   chmod 755 uploads outputs temp logs
   ```

3. **Insufficient memory**
   ```bash
   # Reduce worker thread count
   python main.py --workers 2
   ```

4. **RKNN toolkit issues**
   ```bash
   # Check RKNN toolkit installation
   python -c "from rknn.api import RKNN; print('RKNN toolkit installed successfully')"
   ```

### Debug Mode

```bash
# Enable verbose logging
export PYTHONPATH=.
python -u main.py --debug

# View real-time logs
tail -f logs/server.log
```

## 🚀 Advanced Features

### Historical Task Management
- Automatically save completed task records
- Support for querying historical task status
- Persistent storage of result files

### Automatic Model Analysis
- Intelligent model format recognition
- Automatic grouping of related files
- Model file integrity validation

### Callback Mechanism
- Support for task completion callbacks
- Custom notification URLs
- Status change notifications

## 🤝 Contributing

Welcome to submit Issues and Pull Requests to improve the project.

### Development Environment Setup
```bash
git clone <repository-url>
cd rknn_model_conversion
pip install -r requirements.txt
python -m pytest tests/  # Run tests
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

If you encounter problems or have any questions, please:

1. Check the troubleshooting section of this documentation
2. Search existing Issues
3. Create a new Issue with detailed information

---

**Note**: Please ensure you have properly installed RKNN Toolkit2, which is the core dependency for model conversion.

