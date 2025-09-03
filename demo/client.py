# client_discovery.py - Client service discovery
import socket
import json
import time
import threading
import logging
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServiceDiscoveryClient:
    """Service discovery client"""

    def __init__(self, broadcast_port: int = 9999, response_port: int = 9998):
        """
        Initialize service discovery client

        Args:
            broadcast_port: Target port for sending broadcasts
            response_port: Local port for receiving responses
        """
        self.broadcast_port = broadcast_port
        self.response_port = response_port
        self.discovered_services = []

    def discover_services(
        self,
        service_name: str = "model_conversion",
        timeout: float = 3.0,
        retry_times: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Discover specified services

        Args:
            service_name: Name of service to discover
            timeout: Timeout for waiting for responses
            retry_times: Number of retry attempts

        Returns:
            List of discovered services
        """
        all_services = []

        for attempt in range(retry_times):
            logger.info(f"Discovery attempt {attempt + 1}/{retry_times}")

            # Clear previous results
            self.discovered_services = []

            # Start response listener thread
            listener_thread = threading.Thread(
                target=self._listen_for_responses, args=(timeout,)
            )
            listener_thread.start()

            # Wait for listener to be ready
            time.sleep(0.1)

            # Send broadcast request
            self._send_broadcast_request(service_name)

            # Wait for listener to complete
            listener_thread.join()

            # Collect services discovered in this attempt
            all_services.extend(self.discovered_services)

            if all_services:
                break  # If services found, no need to continue retrying

            if attempt < retry_times - 1:
                logger.info(f"No services found, retrying in 1 second...")
                time.sleep(1)

        # Remove duplicates (based on IP and port)
        unique_services = []
        seen = set()
        for service in all_services:
            key = (service.get("ip"), service.get("port"))
            if key not in seen:
                seen.add(key)
                unique_services.append(service)

        return unique_services

    def _send_broadcast_request(self, service_name: str):
        """Send broadcast request"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        # Prepare request data
        request = {
            "type": "service_discovery",
            "service": service_name,
            "response_port": self.response_port,
            "timestamp": time.time(),
        }

        request_data = json.dumps(request).encode("utf-8")

        # Send to broadcast address
        try:
            # Try multiple broadcast addresses to ensure coverage of different network configurations
            broadcast_addresses = [
                "<broadcast>",  # Default broadcast address
                "255.255.255.255",  # Global broadcast
            ]

            # Get local network broadcast address
            local_broadcast = self._get_local_broadcast_address()
            if local_broadcast:
                broadcast_addresses.append(local_broadcast)

            for broadcast_addr in broadcast_addresses:
                try:
                    sock.sendto(request_data, (broadcast_addr, self.broadcast_port))
                    logger.info(
                        f"Broadcast sent to {broadcast_addr}:{self.broadcast_port}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to send to {broadcast_addr}: {e}")

        finally:
            sock.close()

    def _get_local_broadcast_address(self) -> Optional[str]:
        """Get local network broadcast address"""
        try:
            import ipaddress

            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()

            # Calculate broadcast address (assuming /24 network)
            network = ipaddress.IPv4Network(f"{local_ip}/8", strict=False)
            print(
                f"Local IP: {local_ip}, Network: {network} , Broadcast: {network.broadcast_address}"
            )
            return str(network.broadcast_address)
        except Exception:
            return None

    def _listen_for_responses(self, timeout: float):
        """Listen for service responses"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            sock.bind(("", self.response_port))
            sock.settimeout(0.5)  # Use shorter timeout for more frequent checking

            logger.info(f"Listening for responses on port {self.response_port}")

            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    data, addr = sock.recvfrom(4096)
                    response = json.loads(data.decode("utf-8"))

                    # Verify it's a service announcement
                    if response.get("type") == "service_announcement":
                        response["discovered_from"] = addr[0]
                        self.discovered_services.append(response)
                        logger.info(
                            f"Discovered service at {response.get('ip')}:{response.get('port')}"
                        )

                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"Error processing response: {e}")

        finally:
            sock.close()


def discover_model_service():
    """Convenience function: discover model conversion service"""
    client = ServiceDiscoveryClient()

    logger.info("Searching for model conversion service...")
    services = client.discover_services(
        service_name="model_conversion", timeout=3.0, retry_times=2
    )

    if services:
        logger.info(f"\nFound {len(services)} model conversion service(s):")
        for idx, service in enumerate(services, 1):
            print(f"\n--- Service {idx} ---")
            print(f"IP: {service.get('ip')}")
            print(f"Port: {service.get('port')}")
            print(f"API Endpoint: {service.get('api_endpoint')}")
            print(f"Health Endpoint: {service.get('health_endpoint')}")

            info = service.get("info", {})
            if info:
                print(f"Version: {info.get('version')}")
                print(f"Capabilities: {info.get('capabilities')}")
                print(f"Max Model Size: {info.get('max_model_size')}")

        return services[0]  # Return first discovered service
    else:
        logger.warning("No model conversion service found in the network")
        return None


# Usage example
if __name__ == "__main__":
    # Simple usage
    service = discover_model_service()

    # if service:
    #     # Now you can use the discovered service
    #     import requests

    #     health_url = service.get('health_endpoint')
    #     try:
    #         response = requests.get(health_url, timeout=5)
    #         if response.status_code == 200:
    #             print(f"\n✓ Service is healthy!")
    #         else:
    #             print(f"\n✗ Service returned status code: {response.status_code}")
    #     except Exception as e:
    #         print(f"\n✗ Failed to connect to service: {e}")
