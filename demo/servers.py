# model_service.py - Model conversion server
import socket
import json
import threading
import time
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelServiceDiscovery:
    """Service discovery component for model conversion service"""

    def __init__(
        self,
        service_name: str = "model_conversion_service",
        service_port: int = 8080,
        broadcast_port: int = 9999,
        service_info: Dict[str, Any] = None,
    ):
        """
        Initialize service discovery component

        Args:
            service_name: Service name
            service_port: HTTP port of model conversion service
            broadcast_port: UDP port for listening to broadcasts
            service_info: Additional service information
        """
        self.service_name = service_name
        self.service_port = service_port
        self.broadcast_port = broadcast_port
        self.service_info = service_info or {}
        self.running = False
        self.sock = None

    def get_local_ip(self):
        """Get local IP address"""
        try:
            # Create a UDP socket connection to external address (without actually sending data)
            # This method can get the correct LAN IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception:
            return "127.0.0.1"

    def start_listening(self):
        """Start listening for broadcast requests"""
        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Bind to broadcast port
        self.sock.bind(("", self.broadcast_port))

        logger.info(
            f"Model service discovery listening on UDP port {self.broadcast_port}"
        )

        while self.running:
            try:
                # Set timeout to check running status
                self.sock.settimeout(1.0)
                data, addr = self.sock.recvfrom(1024)

                # Handle received message
                self._handle_discovery_request(data, addr)

            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Error in discovery listener: {e}")

    def _handle_discovery_request(self, data: bytes, addr: tuple):
        """Handle service discovery request"""
        try:
            # Parse request
            request = json.loads(data.decode("utf-8"))
            logger.info(
                f"Received discovery request from {addr[0]}:{addr[1]} - {request}"
            )

            # Check if it's a service discovery request
            if request.get("type") == "service_discovery" and request.get(
                "service"
            ) in ["model_conversion", "all", self.service_name]:

                # Prepare response data
                response = {
                    "type": "service_announcement",
                    "service_name": self.service_name,
                    "ip": self.get_local_ip(),
                    "port": self.service_port,
                    "api_endpoint": f"http://{self.get_local_ip()}:{self.service_port}/api",
                    "health_endpoint": f"http://{self.get_local_ip()}:{self.service_port}/api/health",
                    "timestamp": time.time(),
                    "info": self.service_info,
                }

                # Use specified response port if provided in request
                response_port = request.get("response_port", addr[1])

                # Send response
                response_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                response_data = json.dumps(response).encode("utf-8")
                response_sock.sendto(response_data, (addr[0], response_port))
                response_sock.close()

                logger.info(f"Sent service announcement to {addr[0]}:{response_port}")

        except Exception as e:
            logger.error(f"Error handling discovery request: {e}")

    def stop(self):
        """Stop listening"""
        self.running = False
        if self.sock:
            self.sock.close()
        logger.info("Service discovery stopped")


# Usage example - Integration in model conversion service
def run_model_service():
    """Run model conversion service (including service discovery)"""

    # Create service discovery component
    discovery = ModelServiceDiscovery(
        service_name="model_conversion_service",
        service_port=8080,  # Your model conversion service HTTP port
        broadcast_port=9999,  # UDP broadcast listening port
        service_info={
            "version": "1.0.0",
            "capabilities": ["onnx", "tensorflow", "pytorch"],
            "max_model_size": "2GB",
        },
    )

    # Start service discovery in background thread
    discovery_thread = threading.Thread(target=discovery.start_listening, daemon=True)
    discovery_thread.start()

    # Your model conversion service main logic here
    logger.info("Model conversion service is running...")

    try:
        # Keep service running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        discovery.stop()


if __name__ == "__main__":
    run_model_service()
