"""Server entry point — ``python -m maniparena.launch``."""

import argparse
import logging
import sys

from maniparena.server import WebSocketModelServer


def main():
    parser = argparse.ArgumentParser(
        description="ManipArena model server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--control-mode", type=str, default="end_pose", choices=["joints", "end_pose"])
    parser.add_argument("--action-horizon", type=int, default=50, help="Action sequence length")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Import here so users can swap out the policy module without touching this file.
    # By default we look for examples/my_policy.py on sys.path.
    try:
        from my_policy import MyPolicy
    except ImportError:
        logger.error(
            "Cannot import MyPolicy. Make sure examples/my_policy.py (or your own "
            "policy file) is on PYTHONPATH, e.g.: PYTHONPATH=examples python -m maniparena.launch ..."
        )
        sys.exit(1)

    logger.info(f"checkpoint={args.checkpoint}  mode={args.control_mode}  "
                f"horizon={args.action_horizon}  device={args.device}")

    policy = MyPolicy(
        checkpoint_path=args.checkpoint,
        control_mode=args.control_mode,
        action_horizon=args.action_horizon,
        device=args.device,
    )

    server = WebSocketModelServer(policy=policy, host=args.host, port=args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped (Ctrl+C)")


if __name__ == "__main__":
    main()
