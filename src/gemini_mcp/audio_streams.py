"""Audio streaming handlers for microphone input and speaker output."""

import asyncio
import pyaudio

try:
    from .config import (
        CHUNK_SIZE,
        OUTPUT_CHUNK_SIZE,
        SEND_SAMPLE_RATE,
        RECEIVE_SAMPLE_RATE,
    )
    from .audio_utils import resample_audio
except ImportError:
    from config import (
        CHUNK_SIZE,
        OUTPUT_CHUNK_SIZE,
        SEND_SAMPLE_RATE,
        RECEIVE_SAMPLE_RATE,
    )
    from audio_utils import resample_audio


class AudioStreamHandler:
    """Handles audio streaming for microphone input and speaker output."""

    def __init__(
        self,
        mode,
        format,
        mic_channels,
        speaker_channels,
        mic_index,
        speaker_index,
        mic_sample_rate,
        speaker_sample_rate,
        audio_in_queue,
        out_queue,
        mic_lock,
        audio_stream_lock,
        pya,
        active_muting=True,
    ):
        """
        Initialize audio stream handler.

        Args:
            mode: Operating mode ("sim" or "robot")
            format: Audio format (pyaudio format constant)
            mic_channels: Number of microphone channels
            speaker_channels: Number of speaker channels
            mic_index: Microphone device index
            speaker_index: Speaker device index
            mic_sample_rate: Hardware microphone sample rate
            speaker_sample_rate: Hardware speaker sample rate
            audio_in_queue: Queue for incoming audio from Gemini
            out_queue: Queue for outgoing audio to Gemini
            mic_lock: Lock for microphone muting
            audio_stream_lock: Lock for audio stream state
            pya: PyAudio instance to use for audio I/O
            active_muting: Whether to mute mic during playback
        """
        self.mode = mode
        self.format = format
        self.mic_channels = mic_channels
        self.speaker_channels = speaker_channels
        self.mic_index = mic_index
        self.speaker_index = speaker_index
        self.mic_sample_rate = mic_sample_rate
        self.speaker_sample_rate = speaker_sample_rate
        self.api_sample_rate = SEND_SAMPLE_RATE
        self.api_output_sample_rate = RECEIVE_SAMPLE_RATE
        self.chunk_size = CHUNK_SIZE
        self.received_audio_buffer = OUTPUT_CHUNK_SIZE

        self.audio_in_queue = audio_in_queue
        self.out_queue = out_queue
        self.mic_lock = mic_lock
        self.audio_stream_lock = audio_stream_lock
        self.active_muting = active_muting

        # Control flags
        self.mic_active = True
        self.audio_stream_active = False
        self.last_audio_chunk_time = None

        # PyAudio instance (shared)
        self.pya = pya
        self.audio_stream = None

    async def listen_audio(self):
        """
        Continuously capture audio from microphone and add to output queue.

        Sets up microphone input stream and reads audio data in chunks.
        Resamples audio if hardware rate differs from API rate (robot mode).
        """
        # Get microphone info
        if self.mic_index is not None:
            mic_info = self.pya.get_device_info_by_index(self.mic_index)
        else:
            mic_info = self.pya.get_default_input_device_info()
        print("Microphone:", mic_info["name"])

        # Initialize audio input stream
        self.audio_stream = await asyncio.to_thread(
            self.pya.open,
            format=self.format,
            channels=self.mic_channels,
            rate=self.mic_sample_rate,
            input=True,
            input_device_index=self.mic_index if self.mic_index is not None else mic_info["index"],
            frames_per_buffer=self.chunk_size,
        )

        # Configure overflow handling for debug vs release
        overflow_kwargs = {"exception_on_overflow": False} if __debug__ else {}

        stream_active = True

        # Continuously read audio data
        while True:
            # Check if mic should be active
            async with self.mic_lock:
                mic_currently_active = self.mic_active

            if mic_currently_active:
                # If stream was stopped, restart it
                if not stream_active:
                    await asyncio.to_thread(self.audio_stream.start_stream)
                    stream_active = True

                # Read audio data (blocking call, no need for sleep)
                audio_data = await asyncio.to_thread(
                    self.audio_stream.read, self.chunk_size, **overflow_kwargs
                )

                # Resample if hardware rate differs from API rate (robot mode)
                if self.mic_sample_rate != self.api_sample_rate:
                    audio_data = resample_audio(audio_data, self.mic_sample_rate, self.api_sample_rate)

                await self.out_queue.put({"data": audio_data, "mime_type": "audio/pcm"})
            else:
                # Stop the stream completely to prevent any audio capture
                if stream_active:
                    await asyncio.to_thread(self.audio_stream.stop_stream)
                    stream_active = False

                # Just sleep while muted - no audio is being captured
                await asyncio.sleep(0.1)

    async def play_audio(self):
        """
        Play audio responses from Gemini through speakers.

        Continuously reads audio data from input queue and plays it.
        Mutes microphone during playback to prevent feedback.
        Resamples audio if hardware rate differs from API output rate (robot mode).
        """
        # Initialize audio output stream
        audio_stream = await asyncio.to_thread(
            self.pya.open,
            format=self.format,
            channels=self.speaker_channels,
            rate=self.speaker_sample_rate,
            output=True,
            output_device_index=self.speaker_index,
            frames_per_buffer=self.received_audio_buffer,
        )

        audio_playing = False

        # Continuously play audio from queue
        while True:
            try:
                # Wait for audio with a reasonable timeout
                try:
                    audio_bytes = await asyncio.wait_for(self.audio_in_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Check if we were playing audio and the stream is now complete
                    if audio_playing:
                        async with self.audio_stream_lock:
                            stream_still_active = self.audio_stream_active

                        # If stream is complete and queue is empty, we're done
                        if not stream_still_active and self.audio_in_queue.empty():
                            if self.active_muting:
                                async with self.mic_lock:
                                    self.mic_active = True
                                    audio_playing = False
                                    print("🎤 Microphone unmuted - audio playback complete")
                            else:
                                audio_playing = False
                    continue

                # Update last audio time
                self.last_audio_chunk_time = asyncio.get_event_loop().time()

                # If this is the first audio chunk in a sequence, mute the microphone (if enabled)
                if not audio_playing:
                    if self.active_muting:
                        async with self.mic_lock:
                            self.mic_active = False
                            audio_playing = True
                            print("🔇 Microphone muted while audio is playing")

                        # Small delay to ensure mic is fully muted
                        await asyncio.sleep(0.1)
                    else:
                        audio_playing = True

                # Resample if hardware rate differs from API output rate (robot mode)
                if self.speaker_sample_rate != self.api_output_sample_rate:
                    audio_bytes = resample_audio(audio_bytes, self.api_output_sample_rate, self.speaker_sample_rate)

                # Play the audio
                await asyncio.to_thread(audio_stream.write, audio_bytes)

            except Exception as e:
                print(f"🔴 Audio playback error: {str(e)}")
                # Re-enable microphone in case of error (if muting is enabled)
                if self.active_muting:
                    async with self.mic_lock:
                        self.mic_active = True
                        audio_playing = False
                        print("🎤 Microphone unmuted after audio error")
                else:
                    audio_playing = False
                await asyncio.sleep(0.1)
