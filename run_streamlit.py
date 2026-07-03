import ssl
import sys

# Monkey-patch numpy to support deprecated attributes removed in numpy 1.24+
# This prevents errors in older versions of TensorFlow
import numpy as np
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float

# Monkey-patch ssl to ignore windows store certs which are causing a crash
original_load_default_certs = ssl.SSLContext.load_default_certs
def patched_load_default_certs(self, purpose=ssl.Purpose.SERVER_AUTH):
    try:
        original_load_default_certs(self, purpose)
    except ssl.SSLError:
        pass # Ignore certificate load errors on windows
ssl.SSLContext.load_default_certs = patched_load_default_certs

# Run streamlit
from streamlit.web import cli
if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "app.py"]
    sys.exit(cli.main())
