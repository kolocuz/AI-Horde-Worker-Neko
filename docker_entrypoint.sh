#!/bin/bash
set -euo pipefail

bin/micromamba run -r conda -n linux python generate_docker_bridge_config.py

exec ./horde-bridge.sh "$@"
