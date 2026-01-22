"""Audio utility functions and classes."""

import numpy as np
import pyaudio


# Simple buffer pool to reduce memory allocations
class BufferPool:
    """Simple object pool for numpy arrays to reduce garbage collection pressure"""

    def __init__(self, buffer_size, max_buffers=20):
        self.buffer_size = buffer_size
        self.max_buffers = max_buffers
        self._pool = []

    def get(self):
        """Get a buffer from the pool or create a new one"""
        if self._pool:
            return self._pool.pop()
        return np.empty(self.buffer_size, dtype=np.int16)

    def put(self, buffer):
        """Return a buffer to the pool"""
        if len(self._pool) < self.max_buffers:
            self._pool.append(buffer)


# Audio resampling functions
def resample_audio(audio_data, original_rate, target_rate):
    """Resample audio data from original_rate to target_rate using fast linear interpolation"""
    if original_rate == target_rate:
        return audio_data

    # Convert bytes to numpy array
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    # Calculate resampling ratio
    ratio = target_rate / original_rate

    # Calculate new length
    new_length = int(len(audio_array) * ratio)

    # Fast linear interpolation instead of FFT-based resampling
    # Create indices for the new sample positions
    old_indices = np.arange(len(audio_array))
    new_indices = np.linspace(0, len(audio_array) - 1, new_length)

    # Use numpy's interp for fast linear interpolation
    resampled = np.interp(new_indices, old_indices, audio_array)

    # Convert back to int16
    resampled = resampled.astype(np.int16)

    # Convert back to bytes
    return resampled.tobytes()


def list_audio_devices():
    """
    List all available PyAudio devices for debugging.

    Returns:
        tuple: (default_input_index, default_output_index, all_devices_info)
    """
    pya_temp = pyaudio.PyAudio()
    devices_info = []

    try:
        default_input = pya_temp.get_default_input_device_info()
        default_input_index = default_input['index']
    except Exception:
        default_input_index = None

    try:
        default_output = pya_temp.get_default_output_device_info()
        default_output_index = default_output['index']
    except Exception:
        default_output_index = None

    print("\n" + "="*60)
    print("Available Audio Devices:")
    print("="*60)

    for i in range(pya_temp.get_device_count()):
        try:
            info = pya_temp.get_device_info_by_index(i)
            devices_info.append(info)

            device_type = []
            if info['maxInputChannels'] > 0:
                device_type.append("INPUT")
                if i == default_input_index:
                    device_type.append("(DEFAULT INPUT)")
            if info['maxOutputChannels'] > 0:
                device_type.append("OUTPUT")
                if i == default_output_index:
                    device_type.append("(DEFAULT OUTPUT)")

            type_str = " | ".join(device_type) if device_type else "UNAVAILABLE"

            print(f"[{i}] {info['name']}")
            print(f"    Type: {type_str}")
            print(f"    Channels: In={info['maxInputChannels']}, Out={info['maxOutputChannels']}")
            print(f"    Sample Rate: {int(info['defaultSampleRate'])} Hz")
            print()
        except Exception as e:
            print(f"[{i}] Error reading device: {e}")
            print()

    print("="*60 + "\n")
    pya_temp.terminate()

    return default_input_index, default_output_index, devices_info


def find_audio_device(device_index, device_type="input"):
    """
    Validate and find an audio device, with fallback to default.

    Args:
        device_index: Specific device index to use, or None for default
        device_type: "input" or "output"

    Returns:
        int or None: Valid device index
    """
    pya_temp = pyaudio.PyAudio()

    try:
        # If specific index is requested, validate it
        if device_index is not None:
            try:
                info = pya_temp.get_device_info_by_index(device_index)
                if device_type == "input" and info['maxInputChannels'] > 0:
                    pya_temp.terminate()
                    return device_index
                elif device_type == "output" and info['maxOutputChannels'] > 0:
                    pya_temp.terminate()
                    return device_index
                else:
                    print(f"⚠️  Device {device_index} doesn't support {device_type}, using default instead")
            except Exception as e:
                print(f"⚠️  Device {device_index} not available: {e}")
                print(f"   Falling back to default {device_type} device")

        # Fall back to default device
        if device_type == "input":
            default_info = pya_temp.get_default_input_device_info()
        else:
            default_info = pya_temp.get_default_output_device_info()

        pya_temp.terminate()
        return default_info['index']

    except Exception as e:
        print(f"🔴 Error finding {device_type} device: {e}")
        pya_temp.terminate()
        return None
