"""
Launch script for the Industrial Component Anomaly Detector QA Server.
"""

import sys
from pathlib import Path
import uvicorn

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import settings

if __name__ == "__main__":
    print(f"Starting {settings.system.app_name} v{settings.system.version}...")
    print(f"Web Dashboard: http://localhost:{settings.system.port}")
    print(f"API Documentation: http://localhost:{settings.system.port}/docs")
    uvicorn.run(
        "src.api.app:app",
        host=settings.system.host,
        port=settings.system.port,
        reload=False,
        log_level=settings.system.log_level.lower(),
    )
