import subprocess
import os

# Start the Flask backend
backend = subprocess.Popen(["python", "app.py"])

# Start the React frontend
frontend_dir = os.path.join(os.getcwd(), "fake-news-detection-frontend")
frontend = subprocess.Popen(["npm", "start"], cwd=frontend_dir)

# Wait for both to finish (Ctrl+C to stop)
try:
    backend.wait()
    frontend.wait()
except KeyboardInterrupt:
    backend.terminate()
    frontend.terminate()