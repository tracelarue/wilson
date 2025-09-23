# Wilson Project - UV & pyproject.toml Configuration

This document explains how to use `uv` with the Wilson project's `pyproject.toml` configuration.

## Overview

The Wilson project now includes a comprehensive `pyproject.toml` file that manages all Python dependencies using UV, a fast Python package installer and resolver. This configuration handles dependencies for:

- 🤖 **ROS2 robotics packages** (via system dependencies)
- 🧠 **AI/ML capabilities** (Gemini AI, computer vision)
- 🎥 **Multi-modal interfaces** (audio, video, screen capture)
- 🔧 **Development tools** (testing, linting, formatting)

## Installation

### Prerequisites

1. **Install UV** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # or
   pip install uv
   ```

2. **System Dependencies** (for audio support):
   ```bash
   # Ubuntu/Debian
   sudo apt-get install portaudio19-dev
   
   # macOS
   brew install portaudio
   ```

### Basic Installation

```bash
# Navigate to the Wilson project
cd wilson

# Install core dependencies only
uv sync --no-dev

# Install with development dependencies
uv sync

# Install specific feature sets
uv sync --extra audio      # Audio processing support
uv sync --extra hardware   # Hardware-specific dependencies
uv sync --extra gemini     # Extended Gemini AI features
uv sync --extra full       # All features combined
```

## Usage Examples

### Running Applications

```bash
# Run Gemini Live applications
cd gemini-live
uv run python gemini_live.py --mode=none

cd gemini-live-mcp
uv run python main.py --mode=camera

# Run tests
uv run python test_installation.py
uv run python uv_usage_examples.py
```

### Development Workflow

```bash
# Install with development dependencies
uv sync --group dev

# Run code formatting
uv run black .
uv run isort .

# Run linting
uv run flake8 .
uv run mypy .

# Run tests
uv run pytest
```

## Project Structure

The `pyproject.toml` defines several dependency groups:

### Core Dependencies
- **Computer Vision**: `opencv-python`, `pillow`, `numpy`
- **AI/ML**: `google-genai`, `google-generativeai`
- **System Integration**: `mss`, `python-dotenv`, `pyserial`
- **MCP Support**: `mcp[cli]`
- **Utilities**: `pyyaml`, `transforms3d`

### Optional Dependencies

| Group | Purpose | Install Command |
|-------|---------|-----------------|
| `dev` | Development tools (pytest, black, flake8, mypy) | `uv sync --group dev` |
| `audio` | Audio processing (pyaudio) | `uv sync --extra audio` |
| `hardware` | Hardware-specific (ArducamDepthCamera) | `uv sync --extra hardware` |
| `gemini` | Extended Gemini AI features | `uv sync --extra gemini` |
| `full` | All optional features combined | `uv sync --extra full` |

## Configuration Details

### Python Version Support
- **Minimum**: Python 3.10
- **Tested**: Python 3.10, 3.11, 3.12
- **Compatibility**: Automatic backports for Python < 3.11

### Tool Configurations

The `pyproject.toml` includes configurations for:

- **Black**: Code formatting (88 character line length)
- **isort**: Import sorting (Black-compatible profile)
- **Flake8**: Code linting
- **MyPy**: Type checking
- **Pytest**: Testing framework
- **Coverage**: Code coverage reporting

### Console Scripts

The project defines several console scripts:

```bash
# Available after installation
gemini-live           # Gemini Live application
gemini-live-mcp       # Gemini Live with MCP
gemini-node           # ROS2 Gemini node
multimodal-robot      # Multimodal robot interface
tof-pointcloud        # Depth camera utilities
```

## Workspace Structure

The project supports multiple subpackages:

```
wilson/
├── pyproject.toml          # Main project configuration
├── gemini-live/            # Gemini Live standalone
│   ├── pyproject.toml      # Subproject configuration
│   └── uv.lock            # Locked dependencies
├── gemini-live-mcp/        # Gemini Live with MCP
│   ├── pyproject.toml      # Subproject configuration
│   └── uv.lock            # Locked dependencies
└── src/                    # ROS2 packages
    ├── gemini/             # ROS2 Gemini integration
    ├── depth_cam/          # Camera utilities
    └── wilson/             # Main ROS2 package
```

## Common Commands

### Environment Management
```bash
# Show current environment
uv show

# List installed packages
uv pip list

# Update dependencies
uv sync --upgrade

# Clean cache
uv cache clean
```

### Development Commands
```bash
# Run type checking
uv run mypy src/

# Format code
uv run black src/ gemini-live/ gemini-live-mcp/

# Sort imports
uv run isort src/ gemini-live/ gemini-live-mcp/

# Run all linting
uv run flake8 src/
```

### Testing Commands
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/

# Test specific modules
uv run pytest src/gemini/
```

## Troubleshooting

### Common Issues

1. **PyAudio Installation Fails**
   - Install system audio libraries first
   - Use `uv sync --no-extra audio` to skip audio features
   - Install audio dependencies separately: `uv sync --extra audio`

2. **Hardware Dependencies Fail**
   - Hardware-specific packages may not be available on all platforms
   - Use `uv sync --no-extra hardware` for general development

3. **ROS2 Dependencies**
   - ROS2 packages are managed by the system package manager
   - Install ROS2 separately following ROS2 installation instructions
   - Python ROS2 bindings are not included in PyPI

### Platform-Specific Notes

- **Linux**: Full support for all features
- **macOS**: Audio requires `brew install portaudio`
- **Windows**: Limited hardware support, audio may require additional setup

## Integration with ROS2

While Python dependencies are managed by UV, ROS2 integration requires:

1. **System ROS2 Installation**:
   ```bash
   # Follow ROS2 installation instructions for your platform
   # Source ROS2 setup
   source /opt/ros/humble/setup.bash
   ```

2. **Build ROS2 Packages**:
   ```bash
   # Build Wilson ROS2 packages
   colcon build --packages-select wilson depth_cam gemini
   source install/setup.bash
   ```

3. **Combined Usage**:
   ```bash
   # Use UV environment with ROS2
   source /opt/ros/humble/setup.bash
   source install/setup.bash
   uv run python src/gemini/gemini/MultiModal_robot.py
   ```

## Contributing

When contributing to the project:

1. **Install development dependencies**:
   ```bash
   uv sync --group dev
   ```

2. **Run pre-commit checks**:
   ```bash
   uv run black --check .
   uv run isort --check-only .
   uv run flake8 .
   uv run mypy .
   ```

3. **Test your changes**:
   ```bash
   uv run pytest
   uv run python test_installation.py
   ```

4. **Update dependencies** (if needed):
   ```bash
   # Add new dependency
   uv add package-name
   
   # Add development dependency
   uv add --dev package-name
   
   # Add optional dependency
   uv add --optional audio package-name
   ```

## Migration from pip/requirements.txt

If migrating from a traditional `pip` setup:

1. **Remove old files**:
   ```bash
   rm requirements.txt requirements-dev.txt
   ```

2. **Use UV instead of pip**:
   ```bash
   # Old way
   pip install -r requirements.txt
   
   # New way
   uv sync
   ```

3. **Update CI/CD**:
   ```yaml
   # In GitHub Actions, replace pip with uv
   - name: Install dependencies
     run: |
       curl -LsSf https://astral.sh/uv/install.sh | sh
       uv sync --group dev
   ```

This configuration provides a modern, fast, and reliable dependency management system for the Wilson robotics project while maintaining compatibility with existing ROS2 workflows.