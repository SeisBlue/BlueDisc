"""
Device management module for automatic CUDA/MPS detection and manual device selection.
"""

import torch
import subprocess
from typing import Optional


class DeviceManager:
    """
    Manages device selection with automatic detection of available devices.
    Supports CUDA (NVIDIA GPU), MPS (Apple Silicon), and CPU fallback.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize DeviceManager.

        Args:
            device: Manual device specification. Options:
                   - None or "auto": Auto-detect (CUDA > MPS > CPU)
                   - "cuda": Force CUDA
                   - "mps": Force MPS
                   - "cpu": Force CPU
                   - "cuda:0", "cuda:1", etc: Specific CUDA device
        """
        self.manual_device = device
        self.device = self._detect_device()

    def _detect_device(self) -> torch.device:
        """
        Detect the best available device based on priority:
        1. User manual selection
        2. Auto-detect: CUDA > MPS > CPU

        Returns:
            torch.device: The selected device
        """
        if self.manual_device and self.manual_device.lower() != "auto":
            return self._validate_and_get_device(self.manual_device)

        # Auto-detect: CUDA > MPS > CPU
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _validate_and_get_device(self, device_str: str) -> torch.device:
        """
        Validate device string and return torch.device.

        Args:
            device_str: Device specification string

        Returns:
            torch.device: The validated device

        Raises:
            RuntimeError: If device is not available
        """
        device = torch.device(device_str)

        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is not available. "
                    f"Available devices: {self.get_available_devices()}"
                )
            if device.index is not None:
                if device.index >= torch.cuda.device_count():
                    raise RuntimeError(
                        f"CUDA device {device.index} not found. "
                        f"Available CUDA devices: 0-{torch.cuda.device_count() - 1}"
                    )
        elif device.type == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError(
                    "MPS is not available. "
                    f"Available devices: {self.get_available_devices()}"
                )
        elif device.type == "cpu":
            pass  # CPU always available
        else:
            raise RuntimeError(
                f"Unknown device type: {device_str}. "
                f"Available devices: {self.get_available_devices()}"
            )

        return device

    @staticmethod
    def get_available_devices() -> str:
        """
        Get string describing all available devices.

        Returns:
            str: Description of available devices
        """
        devices = []

        if torch.cuda.is_available():
            cuda_count = torch.cuda.device_count()
            devices.append(f"cuda (available: {cuda_count} device(s))")
        else:
            devices.append("cuda (not available)")

        if torch.backends.mps.is_available():
            devices.append("mps (available)")
        else:
            devices.append("mps (not available)")

        devices.append("cpu (always available)")

        return " | ".join(devices)

    def get_device_info(self) -> str:
        """
        Get detailed information about the selected device.

        Returns:
            str: Device information
        """
        device_str = str(self.device)

        if self.device.type == "cuda":
            try:
                # Get GPU name
                result = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    encoding="utf-8",
                )
                gpu_names = [line.strip() for line in result.strip().split("\n")]

                if self.device.index is not None:
                    gpu_name = gpu_names[self.device.index] if self.device.index < len(gpu_names) else "Unknown"
                    return f"CUDA:{self.device.index} ({gpu_name})"
                else:
                    return f"CUDA ({', '.join(gpu_names)})"
            except Exception as e:
                return f"CUDA (Error getting info: {e})"

        elif self.device.type == "mps":
            return "MPS (Apple Silicon Metal Performance Shaders)"

        else:
            return "CPU"

    def move_to_device(self, obj):
        """
        Move an object (tensor/model) to the selected device.

        Args:
            obj: Object to move (usually a tensor or nn.Module)

        Returns:
            Object on the selected device
        """
        return obj.to(self.device)

    def __str__(self) -> str:
        return str(self.device)

    def __repr__(self) -> str:
        return f"DeviceManager(device={self.device}, info={self.get_device_info()})"


def get_device_manager(device: Optional[str] = None) -> DeviceManager:
    """
    Factory function to create a DeviceManager instance.

    Args:
        device: Manual device specification (see DeviceManager.__init__)

    Returns:
        DeviceManager: Configured device manager instance
    """
    return DeviceManager(device)

