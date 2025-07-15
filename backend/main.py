import subprocess
import time

subprocess.Popen(["python", "rfr.py"])
subprocess.Popen(["python", "crowd_analysis.py"])
subprocess.Popen(["python", "chatbot.py"])

# Keep container alive
while True:
    time.sleep(10)