#!/usr/bin/env python3
"""
Simple HTTPS server for serving mobile client with camera access.
Generates self-signed certificate for local testing.
"""

import http.server
import ssl
import socket
import os
from pathlib import Path
import subprocess

def create_self_signed_cert():
    """Create self-signed certificate for HTTPS."""
    cert_file = "server.crt"
    key_file = "server.key"
    
    if os.path.exists(cert_file) and os.path.exists(key_file):
        print("✅ Certificate files already exist")
        return cert_file, key_file
    
    print("🔐 Creating self-signed certificate...")
    
    # Get local IP
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    # Create certificate
    cmd = [
        "openssl", "req", "-x509", "-newkey", "rsa:4096", 
        "-keyout", key_file, "-out", cert_file, 
        "-days", "365", "-nodes",
        "-subj", f"/CN={local_ip}/O=Vigilant-AI/C=US"
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print("✅ Self-signed certificate created")
        return cert_file, key_file
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create certificate: {e}")
        print("💡 Install OpenSSL: sudo apt-get install openssl")
        return None, None
    except FileNotFoundError:
        print("❌ OpenSSL not found")
        print("💡 Install OpenSSL: sudo apt-get install openssl")
        return None, None

def get_local_ip():
    """Get local IP address."""
    try:
        # Connect to a remote address to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except:
        return "localhost"

def main():
    port = 8443  # HTTPS port
    
    # Create certificate
    cert_file, key_file = create_self_signed_cert()
    
    if not cert_file or not key_file:
        print("❌ Cannot create HTTPS server without certificate")
        print("💡 Try the HTTP + localhost solution instead")
        return
    
    # Create HTTPS server
    server_address = ('0.0.0.0', port)
    httpd = http.server.HTTPServer(server_address, http.server.SimpleHTTPRequestHandler)
    
    # Wrap with SSL
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_file, key_file)
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    
    local_ip = get_local_ip()
    
    print("🚀 HTTPS Server Starting...")
    print(f"📱 Mobile URL: https://{local_ip}:{port}/mobile_camera_client.html")
    print("⚠️  You'll see a security warning - click 'Advanced' → 'Proceed'")
    print("🛑 Press Ctrl+C to stop")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
        httpd.shutdown()

if __name__ == "__main__":
    main()

