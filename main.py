import subprocess
import time
import sys

print("Starting services...")
sys.stdout.flush()

# Start services and redirect their output to the main script's output
p_rfr = subprocess.Popen(["python", "-u", "rfr.py"], stdout=sys.stdout, stderr=sys.stderr)
p_crowd = subprocess.Popen(["python", "-u", "crowd_analysis.py"], stdout=sys.stdout, stderr=sys.stderr)
p_chatbot = subprocess.Popen(["python", "-u", "chatbot.py"], stdout=sys.stdout, stderr=sys.stderr)

print("Services launched.")
sys.stdout.flush()

# Keep container alive and monitor processes
while True:
    time.sleep(10)
    if p_rfr.poll() is not None:
        print("RFR process has terminated.", flush=True)
    if p_crowd.poll() is not None:
        print("Crowd analysis process has terminated.", flush=True)
    if p_chatbot.poll() is not None:
        print("Chatbot process has terminated.", flush=True)