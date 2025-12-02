# 1. Use a lightweight Python base image
FROM python:3.9-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy dependencies first (for caching speed)
COPY requirements.txt .

# 4. Install dependencies
# We install libpcap for Scapy (Packet Sniffing)
RUN apt-get update && apt-get install -y libpcap-dev && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the code
COPY . .

# 6. Expose the Streamlit port
EXPOSE 8501

# 7. Default command: Run the Automation Script
# We need to make sure run.sh uses 'streamlit' directly, not 'venv' inside Docker
CMD ["streamlit", "run", "src/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]