import socket
import json
import threading
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PeerNetwork:
    def __init__(self, host='localhost', port=8000, peers=None):
        self.host = host
        self.port = port
        self.peers = peers or []  # List of (host, port) tuples
        self.server_socket = None
        self.is_running = False
        self.message_handler = None

    def set_message_handler(self, handler):
        """Set callback function to handle incoming messages"""
        self.message_handler = handler

    def start_server(self):
        """Start TCP server to listen for incoming connections"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.is_running = True

            logger.info(f"Server started on {self.host}:{self.port}")

            while self.is_running:
                try:
                    client_socket, addr = self.server_socket.accept()
                    logger.info(f"Connection from {addr}")

                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, addr)
                    )
                    client_thread.daemon = True
                    client_thread.start()

                except socket.error as e:
                    if self.is_running:
                        logger.error(f"Server error: {e}")

        except Exception as e:
            logger.error(f"Failed to start server: {e}")

    def _handle_client(self, client_socket, addr):
        """Handle incoming client connection"""
        try:
            while True:
                data = client_socket.recv(4096)
                if not data:
                    break

                try:
                    message = json.loads(data.decode('utf-8'))
                    logger.info(f"Received message from {addr}: {message['type']}")

                    if self.message_handler:
                        response = self.message_handler(message, addr)
                        if response:
                            client_socket.send(json.dumps(response).encode('utf-8'))

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from {addr}: {e}")

        except Exception as e:
            logger.error(f"Error handling client {addr}: {e}")
        finally:
            client_socket.close()

    def send_message(self, peer_host, peer_port, message):
        """Send message to a specific peer"""
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(10)  # 10 second timeout
            client_socket.connect((peer_host, peer_port))

            json_message = json.dumps(message)
            client_socket.send(json_message.encode('utf-8'))

            # Wait for response
            response_data = client_socket.recv(4096)
            if response_data:
                response = json.loads(response_data.decode('utf-8'))
                logger.info(f"Received response from {peer_host}:{peer_port}")
                return response

        except Exception as e:
            logger.error(f"Failed to send message to {peer_host}:{peer_port}: {e}")
            return None
        finally:
            try:
                client_socket.close()
            except:
                pass

    def broadcast_message(self, message):
        """Send message to all known peers"""
        responses = []
        for peer_host, peer_port in self.peers:
            if peer_host == self.host and peer_port == self.port:
                continue  # Skip self

            response = self.send_message(peer_host, peer_port, message)
            if response:
                responses.append(response)

        return responses

    def stop_server(self):
        """Stop the TCP server"""
        self.is_running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
            logger.info("Server stopped")

    def start_server_thread(self):
        """Start server in a separate thread"""
        server_thread = threading.Thread(target=self.start_server)
        server_thread.daemon = True
        server_thread.start()
        time.sleep(1)  # Give server time to start

    def get_random_peer(self):
        """Get a random peer from the peer list (excluding self)"""
        import random
        available_peers = [
            (host, port) for host, port in self.peers
            if not (host == self.host and port == self.port)
        ]
        if available_peers:
            return random.choice(available_peers)
        return None