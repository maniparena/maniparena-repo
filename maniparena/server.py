"""
Core WebSocket server implementation.

Protocol with robot_client:
1. Send metadata immediately after connection (msgpack).
2. Receive observation messages from client (msgpack).
3. Run policy.infer(obs) and return result (msgpack).
4. Send text error message when inference fails.
"""

import logging
import threading
from typing import Any, Dict

try:
    import msgpack
except ImportError:
    raise ImportError("Please install: pip install msgpack")

# msgpack-numpy is recommended (client may send numpy arrays), but keep it optional:
# if not installed, numpy arrays will arrive as a dict (data/shape) and can be handled in convert_input/utils.
try:
    import msgpack_numpy as m  # type: ignore
    m.patch()
except ImportError:
    m = None

try:
    import websockets
    import websockets.sync.server
    from websockets.exceptions import ConnectionClosed
except ImportError:
    raise ImportError("Please install: pip install websockets")

logger = logging.getLogger(__name__)


class WebSocketModelServer:
    """WebSocket server that handles robot_client connections."""
    
    def __init__(
        self,
        policy: Any,
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        """
        Initialize server.
        
        Args:
            policy: Policy object with infer() method and metadata property.
            host: Server host.
            port: Server port.
        """
        self.policy = policy
        self.host = host
        self.port = port
        self._infer_lock = threading.Lock()
    
    def _handle_client(self, conn: websockets.sync.server.ServerConnection) -> None:
        """Handle a single client connection."""
        client_addr = conn.remote_address
        logger.info(f"Client connected: {client_addr}")
        
        try:
            # 1. Send metadata right after connection
            metadata = getattr(self.policy, "metadata", {}) or {}
            # Use explicit msgpack options to avoid bytes keys/values surprises across environments.
            metadata_bytes = msgpack.packb(metadata, use_bin_type=True)
            conn.send(metadata_bytes)
            logger.info(f"Sent metadata to {client_addr}: {metadata}")
            
            # 2. Handle inference requests in a loop
            while True:
                try:
                    # Receive observation
                    message = conn.recv()
                    
                    # If this is a text message (error/control), log and continue
                    if isinstance(message, str):
                        logger.warning(f"Received text message from {client_addr}: {message}")
                        continue
                    
                    # Decode observation
                    # raw=False ensures str keys (compatible with robot_client's dict access patterns).
                    obs = msgpack.unpackb(message, raw=False)
                    logger.debug(f"Received observation from {client_addr}, keys: {list(obs.keys())}")
                    
                    # Run policy inference
                    try:
                        with self._infer_lock:
                            result = self.policy.infer(obs)
                    except Exception as exc:
                        logger.exception(f"Policy inference error for {client_addr}")
                        # Send error message as text
                        conn.send(f"Error in policy inference: {exc}", text=True)
                        continue
                    
                    # Encode and send result
                    result_bytes = msgpack.packb(result, use_bin_type=True)
                    conn.send(result_bytes)
                    logger.debug(f"Sent result to {client_addr}")
                    
                except ConnectionClosed:
                    logger.info(f"Client disconnected: {client_addr}")
                    break
                    
        except Exception as e:
            logger.exception(f"Unhandled error in client handler for {client_addr}")
        finally:
            logger.info(f"Client handler finished: {client_addr}")
    
    def serve_forever(self) -> None:
        """Start server and run forever."""
        uri = f"ws://{self.host}:{self.port}"
        logger.info(f"Starting WebSocket server on {uri}")
        
        with websockets.sync.server.serve(
            self._handle_client,
            host=self.host,
            port=self.port,
            max_size=None,  # No message size limit
            compression=None,  # Disable compression for performance
        ) as server:
            logger.info("=" * 60)
            logger.info(f"WebSocket server is running on {uri}")
            logger.info("Server is ready and waiting for connections...")
            logger.info("=" * 60)
            server.serve_forever()
