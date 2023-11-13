import os
import sys
from videojitter import _version_generated


def get_version():
    return os.environ.get("VIDEOJITTER_OVERRIDE_VERSION", _version_generated.version)


def print_banner(module_name):
    print(f"{module_name} from videojitter {get_version()}", file=sys.stderr)
