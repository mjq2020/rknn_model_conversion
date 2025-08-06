#!/usr/bin/env python3
"""
RKNN Model Conversion Daemon
Provides network interface service to receive model conversion requests and return conversion results
"""

import asyncio
import signal
import sys
import argparse
from typing import Optional

from utils.config import ServerConfig, ensure_directories
from server.api_server import APIServer
from utils.logger import logger


class RKNNConverterDaemon:
    """RKNN Conversion Daemon"""

    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or ServerConfig()
        self.api_server: Optional[APIServer] = None
        self.running = False

    async def start(self):
        """Start daemon"""
        if self.running:
            logger.warning("Daemon is already running")
            return

        try:
            # Ensure directories exist
            ensure_directories()

            # Create and start API server
            self.api_server = APIServer(self.config)
            await self.api_server.start()

            self.running = True
            logger.info("RKNN conversion daemon started successfully")

            # Keep running
            while self.running:
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Failed to start daemon: {e}")
            await self.stop()
            sys.exit(1)

    async def stop(self):
        """Stop daemon"""
        if not self.running:
            return

        logger.info("Stopping daemon...")
        self.running = False

        if self.api_server:
            await self.api_server.stop()

        logger.info("Daemon stopped")


def signal_handler(signum, frame):
    """Signal handler"""
    logger.info(f"Received signal {signum}, stopping service...")
    asyncio.create_task(daemon.stop())


async def main():
    """Main function"""
    global daemon

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RKNN Model Conversion Daemon")
    parser.add_argument("--host", default="0.0.0.0", help="Server host address")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--workers", type=int, default=4, help="Maximum worker threads")
    parser.add_argument(
        "--upload-dir", default="./uploads", help="Upload file directory"
    )
    parser.add_argument(
        "--output-dir", default="./outputs", help="Output file directory"
    )
    parser.add_argument("--temp-dir", default="./temp", help="Temporary file directory")
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=500 * 1024 * 1024,
        help="Maximum file size (bytes)",
    )

    args = parser.parse_args()

    # Create configuration
    config = ServerConfig(
        host=args.host,
        port=args.port,
        max_workers=args.workers,
        upload_folder=args.upload_dir,
        output_folder=args.output_dir,
        temp_folder=args.temp_dir,
        max_file_size=args.max_file_size,
    )

    # Create daemon
    daemon = RKNNConverterDaemon(config)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start daemon
        await daemon.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await daemon.stop()


if __name__ == "__main__":
    # Global variable
    daemon: Optional[RKNNConverterDaemon] = None

    # Run main function
    asyncio.run(main())
