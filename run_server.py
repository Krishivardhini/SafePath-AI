"""
SafePath Universal Server Runner.
Run this script from anywhere (VS Code terminal, root directory, or inner directory).
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_INNER = os.path.join(_ROOT, "safepath")

for p in [_ROOT, _INNER]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from safepath.server import run_server
except ImportError:
    from server import run_server

if __name__ == "__main__":
    run_server()
