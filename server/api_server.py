import asyncio
import json
import os
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from aiohttp import web
import aiofiles

from utils.config import (
    ConversionTask,
    RKNNConverterConfig,
    DEFAULT_SERVER_CONFIG,
    ServerConfig,
    ensure_directories,
)
from task_manager import task_manager, TaskStatus
from utils.logger import logger
from __version__ import version

# Import model analyzer, handle possible import errors
try:
    from utils.model_analyzer import model_analyzer

    MODEL_ANALYZER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Model analyzer import failed: {e}")
    MODEL_ANALYZER_AVAILABLE = False
    model_analyzer = None


class APIServer:
    """API Server"""

    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or DEFAULT_SERVER_CONFIG
        # Set client max request size to support large file uploads
        self.app = web.Application(client_max_size=self.config.max_file_size)
        self.setup_routes()
        ensure_directories()

    def setup_routes(self):
        """Setup routes"""
        # Health check
        self.app.router.add_get("/health", self.health_check)

        # Task management
        self.app.router.add_post("/api/tasks", self.create_task)
        self.app.router.add_get("/api/tasks", self.list_tasks)
        self.app.router.add_get("/api/tasks/{task_id}", self.get_task)
        self.app.router.add_delete("/api/tasks/{task_id}", self.cancel_task)

        # File upload
        self.app.router.add_post("/api/upload", self.upload_file)

        # Upload and create task
        self.app.router.add_post(
            "/api/upload_and_create_task", self.upload_and_create_task
        )

        # File download
        self.app.router.add_get("/api/download/{task_id}", self.download_result)

        # Task logs
        self.app.router.add_get("/api/tasks/{task_id}/logs", self.get_task_logs)

        # Static file service
        self.app.router.add_static("/static", path="./static", name="static")

    async def upload_and_create_task(self, request: web.Request) -> web.Response:
        """Upload and create task (supports multi-file models)"""
        try:
            # Check if it's a multipart request
            if (
                not request.content_type
                or "multipart/form-data" not in request.content_type
            ):
                return web.json_response(
                    {"error": "Please use multipart/form-data format to upload files"},
                    status=400,
                )

            reader = await request.multipart()

            uploaded_files = []
            config_data = {}
            task_data = {}

            async for field in reader:
                if "file" in field.name and field.filename:
                    # Validate file extension
                    _, ext = os.path.splitext(field.filename)
                    print(field.name, field.filename, field)
                    print("=" * 100)
                    if ext.lower() not in self.config.allowed_extensions:
                        return web.json_response(
                            {"error": f"Unsupported file type: {ext}"}, status=400
                        )

                    # Generate unique filename
                    filename = f"{uuid.uuid4()}{ext}"
                    file_path = os.path.join(self.config.upload_folder, filename)

                    # Save file
                    size = 0
                    async with aiofiles.open(file_path, "wb") as f:
                        while True:
                            chunk = await field.read_chunk(8192)
                            if not chunk:
                                break
                            await f.write(chunk)
                            size += len(chunk)

                            # Check file size limit
                            if size > self.config.max_file_size:
                                await f.close()
                                os.remove(file_path)
                                return web.json_response(
                                    {
                                        "error": f"File size exceeds limit: {self.config.max_file_size / 1024 / 1024}MB"
                                    },
                                    status=400,
                                )

                    uploaded_files.append(
                        {
                            "original_name": field.filename,
                            "saved_name": filename,
                            "path": file_path,
                            "size": size,
                        }
                    )
                elif field.name and not field.filename:
                    # Process form fields
                    field_value = await field.text()
                    if field.name == "config":
                        try:
                            print("<<<<<<<<", field_value)
                            config_data = json.loads(field_value) if field_value else {}
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Config parameter JSON parsing failed, using default config: {field_value}"
                            )
                            config_data = {}
                    else:
                        task_data[field.name] = field_value

            if not uploaded_files:
                return web.json_response(
                    {"error": "No uploaded files found"}, status=400
                )

            logger.info(f"File upload successful: {len(uploaded_files)} files")

            # Analyze uploaded files and organize into model file groups
            if not MODEL_ANALYZER_AVAILABLE:
                return web.json_response(
                    {
                        "error": "Model analyzer not installed, cannot identify model files"
                    },
                    status=500,
                )

            try:
                model_groups = model_analyzer.analyze_uploaded_files(uploaded_files)
                if not model_groups:
                    return web.json_response(
                        {"error": "Cannot identify valid model file format"}, status=400
                    )

                # If there are multiple model groups, select the first one (batch conversion can be supported later)
                model_files = model_groups[0]

                # Validate model file integrity
                is_valid, error_msg = model_analyzer.validate_model_files(model_files)
                if not is_valid:
                    return web.json_response(
                        {"error": f"Model file validation failed: {error_msg}"},
                        status=400,
                    )

                logger.info(
                    f"Identified model type: {model_files.model_type}, primary file: {os.path.basename(model_files.primary_file)}, auxiliary files: {len(model_files.secondary_files)} files"
                )

            except Exception as e:
                logger.error(f"Model file analysis failed: {e}")
                return web.json_response(
                    {"error": f"Model file analysis failed: {str(e)}"}, status=500
                )

            # Create conversion config (merge user config with default config)
            try:
                print("<<<<<<<<", config_data)
                config = RKNNConverterConfig()
                config.update_config(config_data)
            except Exception as e:
                logger.warning(f"Invalid config parameters, using default config: {e}")
                config = RKNNConverterConfig()

            # Generate task ID
            task_id = task_data.get("task_id", str(uuid.uuid4()))
            print(">>>>>>>>>>", model_files, model_groups)
            # Create task (using model file group)
            task = ConversionTask(
                task_id=task_id,
                model_files=model_files,
                config=config,
                callback_url=task_data.get("callback_url"),
                priority=int(task_data.get("priority", 0)),
                metadata={
                    "model_type": model_files.model_type,
                    "uploaded_files_count": len(uploaded_files),
                    "primary_file": os.path.basename(model_files.primary_file),
                    "secondary_files": [
                        os.path.basename(f) for f in model_files.secondary_files
                    ],
                    "additional_files": [
                        os.path.basename(f) for f in model_files.additional_files
                    ],
                    **(
                        json.loads(task_data.get("metadata", "{}"))
                        if task_data.get("metadata")
                        else {}
                    ),
                },
            )

            # Add to task manager
            final_task_id = task_manager.add_task(task)

            logger.info(f"Task created successfully: {final_task_id}")

            return web.json_response(
                {
                    "task_id": final_task_id,
                    "status": "created",
                    "message": "Task created successfully",
                    "model_type": model_files.model_type,
                    "files_info": {
                        "primary_file": os.path.basename(model_files.primary_file),
                        "secondary_files": [
                            os.path.basename(f) for f in model_files.secondary_files
                        ],
                        "total_files": len(uploaded_files),
                    },
                    "output_path": task.get_output_path(self.config.output_folder),
                }
            )

        except Exception as e:
            logger.error(f"Task creation failed: {e}")
            return web.json_response(
                {"error": f"Task creation failed: {str(e)}"}, status=500
            )

    async def health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint"""
        return web.json_response(
            {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": version,
            }
        )

    async def create_task(self, request: web.Request) -> web.Response:
        """Create conversion task"""
        try:
            # Parse request data
            data = await request.json()

            # Validate required fields
            required_fields = ["model_path"]
            for field in required_fields:
                if field not in data:
                    return web.json_response(
                        {"error": f"Missing required field: {field}"}, status=400
                    )

            # Validate if model file exists
            if not os.path.exists(data["model_path"]):
                return web.json_response(
                    {"error": f'Model file does not exist: {data["model_path"]}'},
                    status=400,
                )

            # Create conversion config (merge user config with default config)
            config_data = data.get("config", {})
            try:
                config = RKNNConverterConfig(**config_data)
            except Exception as e:
                logger.warning(f"Invalid config parameters, using default config: {e}")
                config = RKNNConverterConfig()

            # Generate task ID
            task_id = data.get("task_id", str(uuid.uuid4()))

            # Create task (output_path is optional, will be auto-generated by task)
            task = ConversionTask(
                task_id=task_id,
                model_path=data["model_path"],
                config=config,
                output_path=data.get("output_path"),  # Optional field
                callback_url=data.get("callback_url"),
                priority=data.get("priority", 0),
                metadata=data.get("metadata", {}),
            )

            # Add to task manager
            final_task_id = task_manager.add_task(task)

            logger.info(f"Task created successfully: {final_task_id}")

            return web.json_response(
                {
                    "task_id": final_task_id,
                    "status": "created",
                    "message": "Task created successfully",
                    "output_path": task.get_output_path(self.config.output_folder),
                }
            )

        except Exception as e:
            logger.error(f"Task creation failed: {e}")
            return web.json_response(
                {"error": f"Task creation failed: {str(e)}"}, status=500
            )

    async def list_tasks(self, request: web.Request) -> web.Response:
        """Get task list"""
        try:
            tasks = task_manager.get_all_tasks()

            task_list = []
            for task_info in tasks:
                task_data = {
                    "task_id": task_info.task_id,
                    "status": task_info.status.value,
                    "created_at": (
                        task_info.created_at.isoformat()
                        if task_info.created_at
                        else None
                    ),
                    "started_at": (
                        task_info.started_at.isoformat()
                        if task_info.started_at
                        else None
                    ),
                    "completed_at": (
                        task_info.completed_at.isoformat()
                        if task_info.completed_at
                        else None
                    ),
                    "progress": task_info.progress,
                    "error_message": task_info.error_message,
                    "result_path": task_info.result_path,
                    "is_historical": getattr(task_info, "is_historical", False),
                }

                # For historical tasks, add additional information
                if getattr(task_info, "is_historical", False):
                    task_data["model_name"] = task_info.task.model_path.replace(
                        "<Historical Task-", ""
                    ).replace(">", "")

                task_list.append(task_data)

            return web.json_response({"tasks": task_list, "total": len(task_list)})

        except Exception as e:
            logger.error(f"Failed to get task list: {e}")
            return web.json_response(
                {"error": f"Failed to get task list: {str(e)}"}, status=500
            )

    async def get_task(self, request: web.Request) -> web.Response:
        """Get task details"""
        try:
            task_id = request.match_info["task_id"]
            task_info = task_manager.get_task(task_id)

            if not task_info:
                return web.json_response(
                    {"error": f"Task does not exist: {task_id}"}, status=404
                )

            task_data = {
                "task_id": task_info.task_id,
                "status": task_info.status.value,
                "created_at": (
                    task_info.created_at.isoformat() if task_info.created_at else None
                ),
                "started_at": (
                    task_info.started_at.isoformat() if task_info.started_at else None
                ),
                "completed_at": (
                    task_info.completed_at.isoformat()
                    if task_info.completed_at
                    else None
                ),
                "progress": task_info.progress,
                "error_message": task_info.error_message,
                "result_path": task_info.result_path,
                "is_historical": getattr(task_info, "is_historical", False),
            }

            # For non-historical tasks, add complete metadata
            if not getattr(task_info, "is_historical", False):
                task_data["metadata"] = task_info.task.metadata
            else:
                # For historical tasks, add inferred information
                task_data["model_name"] = task_info.task.model_path.replace(
                    "<Historical Task-", ""
                ).replace(">", "")
                task_data["metadata"] = {
                    "note": "This is a historical task completed before program restart"
                }

            return web.json_response(task_data)

        except Exception as e:
            logger.error(f"Failed to get task details: {e}")
            return web.json_response(
                {"error": f"Failed to get task details: {str(e)}"}, status=500
            )

    async def cancel_task(self, request: web.Request) -> web.Response:
        """Cancel task"""
        try:
            task_id = request.match_info["task_id"]
            success = task_manager.cancel_task(task_id)

            if success:
                return web.json_response(
                    {"message": f"Task {task_id} has been cancelled"}
                )
            else:
                return web.json_response(
                    {"error": f"Failed to cancel task: {task_id}"}, status=400
                )

        except Exception as e:
            logger.error(f"Failed to cancel task: {e}")
            return web.json_response(
                {"error": f"Failed to cancel task: {str(e)}"}, status=500
            )

    async def upload_file(self, request: web.Request) -> web.Response:
        """File upload endpoint"""
        try:
            # Check if it's a multipart request
            if (
                not request.content_type
                or "multipart/form-data" not in request.content_type
            ):
                return web.json_response(
                    {"error": "Please use multipart/form-data format to upload files"},
                    status=400,
                )

            # Parse multipart data
            reader = await request.multipart()

            uploaded_files = []

            async for field in reader:
                if field.name == "file" and field.filename:
                    # Validate file extension
                    _, ext = os.path.splitext(field.filename)
                    if ext.lower() not in self.config.allowed_extensions:
                        return web.json_response(
                            {"error": f"Unsupported file type: {ext}"}, status=400
                        )

                    # Generate unique filename
                    filename = f"{uuid.uuid4()}{ext}"
                    file_path = os.path.join(self.config.upload_folder, filename)

                    # Save file
                    size = 0
                    async with aiofiles.open(file_path, "wb") as f:
                        while True:
                            chunk = await field.read_chunk(8192)
                            if not chunk:
                                break
                            await f.write(chunk)
                            size += len(chunk)

                            # Check file size limit
                            if size > self.config.max_file_size:
                                await f.close()
                                os.remove(file_path)
                                return web.json_response(
                                    {
                                        "error": f"File size exceeds limit: {self.config.max_file_size / 1024 / 1024}MB"
                                    },
                                    status=400,
                                )

                    uploaded_files.append(
                        {
                            "original_name": field.filename,
                            "saved_name": filename,
                            "path": file_path,
                            "size": size,
                        }
                    )

            if not uploaded_files:
                return web.json_response(
                    {"error": "No uploaded files found"}, status=400
                )

            logger.info(f"File upload successful: {len(uploaded_files)} files")

            return web.json_response(
                {"message": "File upload successful", "files": uploaded_files}
            )

        except Exception as e:
            logger.error(f"File upload failed: {e}")
            return web.json_response(
                {"error": f"File upload failed: {str(e)}"}, status=500
            )

    async def download_result(self, request: web.Request) -> web.Response:
        """Download conversion result"""
        try:
            task_id = request.match_info["task_id"]
            task_info = task_manager.get_task(task_id)

            if not task_info:
                return web.json_response(
                    {"error": f"Task does not exist: {task_id}"}, status=404
                )

            if task_info.status != TaskStatus.COMPLETED:
                return web.json_response(
                    {
                        "error": f"Task not completed yet, current status: {task_info.status.value}"
                    },
                    status=400,
                )

            if not task_info.result_path or not os.path.exists(task_info.result_path):
                return web.json_response(
                    {"error": "Conversion result file does not exist"}, status=404
                )

            # Return file
            return web.FileResponse(
                path=task_info.result_path,
                # filename=os.path.basename(task_info.result_path)
            )

        except Exception as e:
            logger.error(f"File download failed: {e}")
            return web.json_response(
                {"error": f"File download failed: {str(e)}"}, status=500
            )

    async def get_task_logs(self, request: web.Request) -> web.Response:
        """Get task logs"""
        try:
            task_id = request.match_info["task_id"]
            task_info = task_manager.get_task(task_id)

            if not task_info:
                return web.json_response(
                    {"error": f"Task does not exist: {task_id}"}, status=404
                )

            return web.json_response({"task_id": task_id, "logs": task_info.logs})

        except Exception as e:
            logger.error(f"Failed to get task logs: {e}")
            return web.json_response(
                {"error": f"Failed to get task logs: {str(e)}"}, status=500
            )

    async def start(self):
        """Start server"""
        logger.info(f"Starting API server: {self.config.host}:{self.config.port}")

        # Set task manager output directory
        task_manager.set_output_folder(self.config.output_folder)

        # Start task manager
        await task_manager.start()

        # Start web server
        runner = web.AppRunner(self.app)
        await runner.setup()

        site = web.TCPSite(runner, self.config.host, self.config.port)
        await site.start()

        logger.info(f"API server started: http://{self.config.host}:{self.config.port}")

        return runner

    async def stop(self):
        """Stop server"""
        logger.info("Stopping API server...")

        # Stop task manager
        await task_manager.stop()

        logger.info("API server stopped")
