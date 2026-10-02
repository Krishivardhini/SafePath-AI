"""
SafePath Universal Server Runner.
Run this script from anywhere (VS Code terminal, root directory, or inner directory).
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_ROOT)

for p in [_ROOT, _PARENT]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from safepath.server import run_server
except ImportError:
    from server import run_server

if __name__ == "__main__":
    run_server()
