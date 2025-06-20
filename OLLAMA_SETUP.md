# Ollama Multi-Agent Setup Guide

This guide explains how to set up and run the knowledge graph alignment system using multiple Ollama servers with `gemma3:1b` models.

## Overview

The system now uses **6 separate Ollama servers**, each running on a different port to handle different entity categories concurrently:

| Agent Category | Port  | Purpose |
|---------------|-------|---------|
| Person        | 11434 | Align person entities |
| Place         | 11435 | Align place/location entities |
| Event         | 11436 | Align event entities |
| BuildingPlace | 11437 | Align building/structure entities |
| CreativeWork  | 11438 | Align creative work entities |
| Uncertain     | 11439 | Align uncertain/ambiguous entities |

## Prerequisites

1. **Install Ollama**: Download and install Ollama from [https://ollama.ai](https://ollama.ai)
2. **Python Dependencies**: All required packages are in `mvp-kg-alignment/pyproject.toml`

## Setup Steps

### 1. Install Dependencies
```bash
cd mvp-kg-alignment
pip install -e .
```

### 2. Pull the Gemma3:1b Model
First, pull the model for all servers:
```bash
python start_ollama_servers.py pull
```

### 3. Start All Ollama Servers
Start all 6 specialized Ollama servers:
```bash
python start_ollama_servers.py start
```

This will:
- Start 6 Ollama servers on ports 11434-11439
- Each server will serve the `gemma3:1b` model
- Keep running until you press Ctrl+C

### 4. Check Server Status
To verify all servers are running:
```bash
python start_ollama_servers.py status
```

### 5. Run the Multi-Agent System
With all Ollama servers running, execute the main alignment script:
```bash
python multi-agent-architecture.py
```

### 6. Stop All Servers
When done, stop all servers:
```bash
python start_ollama_servers.py stop
```
Or press Ctrl+C if you're running `start` command.

## How It Works

1. **OrchestratorAgent** loads and classifies entities
2. **ClassifierAgent** categorizes entities using `gemma3:4b` on port 11434
3. **SpecializedAgents** perform alignment using `gemma3:1b` on their assigned ports:
   - Each agent connects to its own Ollama server
   - Agents run concurrently without interfering with each other
   - Each agent uses category-specific prompts for better alignment accuracy

## Configuration

### Port Configuration
Port assignments are defined in both:
- `start_ollama_servers.py` (server management)
- `multi-agent-architecture.py` (agent configuration)

### Model Configuration
- **Classification**: `gemma3:4b` on port 11434
- **Alignment**: `gemma3:1b` on ports 11435-11439

## Troubleshooting

### Server Not Starting
```bash
# Check if port is already in use
netstat -an | grep :11434

# Kill process using the port (if needed)
lsof -ti:11434 | xargs kill -9
```

### Model Not Found
```bash
# Manually pull model for specific port
OLLAMA_HOST=localhost:11434 ollama pull gemma3:1b
```

### Connection Issues
- Ensure all servers are running: `python start_ollama_servers.py status`
- Check firewall settings
- Verify Ollama installation: `ollama --version`

## Performance Notes

- **Memory Usage**: Running 6 models simultaneously requires significant RAM (estimate: 6-8GB+)
- **Concurrent Processing**: All specialized agents can process different entity batches simultaneously
- **Scalability**: You can adjust the number of worker threads in `OrchestratorAgent` (currently `os.cpu_count() * 2`)

## File Structure
```
├── multi-agent-architecture.py    # Main multi-agent system
├── start_ollama_servers.py        # Server management script
├── OLLAMA_SETUP.md                # This guide
└── mvp-kg-alignment/
    ├── pyproject.toml              # Python dependencies
    ├── categories/                 # Pre-classified entity data
    └── processed_data/             # Output directory
``` 