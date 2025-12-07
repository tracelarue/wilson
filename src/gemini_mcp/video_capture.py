"""Video capture functionality for camera and screen."""

import asyncio
import base64
import io
import cv2
import mss
import mss.tools
import PIL.Image


class VideoCaptureHandler:
    """Handles camera and screen capture for Gemini Live."""

    def __init__(self, out_queue):
        """
        Initialize video capture handler.

        Args:
            out_queue: Queue for sending captured frames
        """
        self.out_queue = out_queue

    def _get_frame(self, cap):
        """
        Capture and process a single frame from camera.

        Args:
            cap: OpenCV VideoCapture object

        Returns:
            dict: Frame data with mime_type and base64-encoded image data, or None if failed
        """
        # Capture frame from camera
        ret, frame = cap.read()
        if not ret:
            return None

        # Convert BGR (OpenCV) to RGB (PIL) to prevent blue tint
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create PIL image and resize for efficiency
        img = PIL.Image.fromarray(frame_rgb)
        img.thumbnail([1024, 1024])

        # Convert to JPEG format
        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        # Return as base64-encoded data
        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        """
        Continuously capture frames from camera and add them to output queue.

        Uses asyncio.to_thread to prevent blocking the audio pipeline.
        Captures frames at 1 second intervals.
        """
        # Initialize camera (0 = default camera)
        # Run in thread to prevent blocking audio pipeline
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)

        # Set lower resolution for faster capture and less bandwidth
        await asyncio.to_thread(cap.set, cv2.CAP_PROP_FRAME_WIDTH, 640)
        await asyncio.to_thread(cap.set, cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            # Capture frame in separate thread
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break

            # Try to add frame without blocking if queue is full (skip frame)
            try:
                self.out_queue.put_nowait(frame)
            except asyncio.QueueFull:
                pass  # Skip this frame if queue is full

            # Send frame at 1 second intervals
            await asyncio.sleep(1.0)

        # Clean up camera resource
        cap.release()

    def _get_screen(self):
        """
        Capture and process a screenshot from the primary monitor.

        Returns:
            dict: Screen data with mime_type and base64-encoded image data
        """
        # Initialize screen capture
        screen_capture = mss.mss()
        primary_monitor = screen_capture.monitors[0]

        # Capture screenshot
        screenshot = screen_capture.grab(primary_monitor)

        # Convert to PIL Image
        image_bytes = mss.tools.to_png(screenshot.rgb, screenshot.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        # Convert to JPEG format
        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        # Return as base64-encoded data
        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):
        """
        Continuously capture screenshots and add them to output queue.

        Captures screenshots at 1 second intervals.
        """
        while True:
            # Capture screenshot in separate thread
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break

            # Try to add frame without blocking if queue is full (skip frame)
            try:
                self.out_queue.put_nowait(frame)
            except asyncio.QueueFull:
                pass  # Skip this frame if queue is full

            # Send screenshot at 1 second intervals
            await asyncio.sleep(1.0)
