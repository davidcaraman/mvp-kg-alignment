#!/usr/bin/env python3
"""
Helper script to start multiple Ollama servers for specialized agents.
Each specialized agent needs its own Ollama server running on a different port.

Port assignments:
- Person Agent: 11434 (default)
- Place Agent: 11435
- Event Agent: 11436
- BuildingPlace Agent: 11437
- CreativeWork Agent: 11438
- Uncertain Agent: 11439
"""

import subprocess
import time
import requests
import os
import signal
import sys
from typing import List, Dict

# Configuration
OLLAMA_MODEL = "gemma3:1b"
AGENT_PORTS = {
    "Person": 11434,
    "Place": 11435,
    "Event": 11436,
    "BuildingPlace": 11437,
    "CreativeWork": 11438,
    "Uncertain": 11439
}

class OllamaServerManager:
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.running_ports: List[int] = []
        
    def check_model_availability(self, port: int) -> bool:
        """Check if the gemma3:1b model is available on the specified port."""
        try:
            response = requests.get(f"http://localhost:{port}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return any(OLLAMA_MODEL in model.get('name', '') for model in models)
        except:
            pass
        return False
    
    def pull_model_if_needed(self, port: int) -> bool:
        """Pull the gemma3:1b model if it's not available."""
        if self.check_model_availability(port):
            print(f"✅ Model {OLLAMA_MODEL} already available on port {port}")
            return True
            
        print(f"📥 Pulling model {OLLAMA_MODEL} on port {port}...")
        try:
            # Pull model using ollama CLI with specific port
            env = os.environ.copy()
            env['OLLAMA_HOST'] = f"localhost:{port}"
            
            result = subprocess.run([
                "ollama", "pull", OLLAMA_MODEL
            ], env=env, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print(f"✅ Successfully pulled {OLLAMA_MODEL} on port {port}")
                return True
            else:
                print(f"❌ Failed to pull {OLLAMA_MODEL} on port {port}: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout while pulling {OLLAMA_MODEL} on port {port}")
            return False
        except Exception as e:
            print(f"❌ Error pulling {OLLAMA_MODEL} on port {port}: {e}")
            return False
    
    def start_ollama_server(self, port: int, agent_name: str) -> bool:
        """Start Ollama server on specified port."""
        print(f"🚀 Starting Ollama server for {agent_name} on port {port}...")
        
        try:
            # Set environment variables for the server
            env = os.environ.copy()
            env['OLLAMA_HOST'] = f"localhost:{port}"
            
            # Start Ollama server
            process = subprocess.Popen([
                "ollama", "serve"
            ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait a bit for server to start
            time.sleep(3)
            
            # Check if server is running
            if self.check_server_health(port):
                self.processes.append(process)
                self.running_ports.append(port)
                print(f"✅ Ollama server for {agent_name} started successfully on port {port}")
                return True
            else:
                process.terminate()
                print(f"❌ Failed to start Ollama server for {agent_name} on port {port}")
                return False
                
        except Exception as e:
            print(f"❌ Error starting server for {agent_name} on port {port}: {e}")
            return False
    
    def check_server_health(self, port: int) -> bool:
        """Check if Ollama server is healthy on specified port."""
        try:
            response = requests.get(f"http://localhost:{port}/api/tags", timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def start_all_servers(self) -> bool:
        """Start all Ollama servers for specialized agents."""
        print("🔄 Starting all Ollama servers for specialized agents...\n")
        
        success_count = 0
        for agent_name, port in AGENT_PORTS.items():
            if self.start_ollama_server(port, agent_name):
                success_count += 1
            print()  # Empty line for readability
        
        print(f"📊 Started {success_count}/{len(AGENT_PORTS)} servers successfully")
        
        if success_count == len(AGENT_PORTS):
            print("🎉 All servers started successfully!")
            return True
        else:
            print("⚠️  Some servers failed to start. Check the logs above.")
            return False
    
    def pull_all_models(self) -> bool:
        """Pull the gemma3:1b model for all ports."""
        print("📥 Pulling models for all servers...\n")
        
        success_count = 0
        for agent_name, port in AGENT_PORTS.items():
            print(f"Processing {agent_name} (port {port})...")
            if self.pull_model_if_needed(port):
                success_count += 1
            print()
        
        print(f"📊 Models ready on {success_count}/{len(AGENT_PORTS)} servers")
        return success_count == len(AGENT_PORTS)
    
    def stop_all_servers(self):
        """Stop all running Ollama servers."""
        print("🛑 Stopping all Ollama servers...")
        
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        
        self.processes.clear()
        self.running_ports.clear()
        print("✅ All servers stopped")
    
    def status(self):
        """Show status of all servers."""
        print("📊 Server Status:")
        print("-" * 50)
        
        for agent_name, port in AGENT_PORTS.items():
            if self.check_server_health(port):
                model_available = "✅" if self.check_model_availability(port) else "❌"
                print(f"{agent_name:15} | Port {port} | Running ✅ | Model {model_available}")
            else:
                print(f"{agent_name:15} | Port {port} | Stopped ❌ | Model ❌")
        print("-" * 50)

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print("\n🛑 Received interrupt signal. Stopping servers...")
    if 'manager' in globals():
        manager.stop_all_servers()
    sys.exit(0)

def main():
    global manager
    manager = OllamaServerManager()
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🤖 Ollama Multi-Server Manager for Specialized Agents")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python start_ollama_servers.py start    # Start all servers")
        print("  python start_ollama_servers.py pull     # Pull models for all servers")
        print("  python start_ollama_servers.py status   # Check server status")
        print("  python start_ollama_servers.py stop     # Stop all servers")
        return
    
    command = sys.argv[1].lower()
    
    if command == "start":
        if manager.start_all_servers():
            print("\n🔄 Servers are running. Press Ctrl+C to stop all servers.")
            try:
                # Keep the script running
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        else:
            print("❌ Failed to start some servers. Exiting.")
            sys.exit(1)
    
    elif command == "pull":
        if manager.pull_all_models():
            print("✅ All models pulled successfully!")
        else:
            print("❌ Failed to pull some models.")
            sys.exit(1)
    
    elif command == "status":
        manager.status()
    
    elif command == "stop":
        manager.stop_all_servers()
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: start, pull, status, stop")
        sys.exit(1)

if __name__ == "__main__":
    main() 