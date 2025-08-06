"""
RKNN Model Conversion Service Version Information
"""

version = "1.1.0"
build_date = "2025-08-06"
author = "Seeed Studio"


def get_version_info():
    """Get detailed version information"""
    return {
        "version": version,
        "build_date": build_date,
        "author": author,
        "features": [
            "Model Conversion Service",
            "Task Queue Management",
            "Historical Task Query",
            "File Upload/Download",
            "RESTful API",
        ],
    }
