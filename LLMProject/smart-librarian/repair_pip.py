import sys, ensurepip, subprocess

# 1) The basic pip installation
ensurepip.bootstrap()

# 2) pip / setuptools / wheel refreshing to stable versions
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade",
                       "pip", "setuptools", "wheel"])

print("✅ pip repaired and upgraded for:", sys.executable)
