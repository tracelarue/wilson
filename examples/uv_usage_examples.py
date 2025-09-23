#!/usr/bin/env python3
"""
Usage examples for the Wilson project with uv.
This demonstrates how to use the dependencies configured in pyproject.toml.
"""

import cv2
import numpy as np
from PIL import Image
import mss
from dotenv import load_dotenv
import os

def screen_capture_example():
    """Example of screen capture functionality."""
    print("📸 Screen Capture Example")
    print("-" * 30)
    
    try:
        with mss.mss() as sct:
            # Get information about the first monitor
            monitor = sct.monitors[1]  # 0 is all monitors, 1 is first monitor
            print(f"Monitor dimensions: {monitor['width']}x{monitor['height']}")
            
            # Take a screenshot
            screenshot = sct.grab(monitor)
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            print(f"Screenshot captured: {img.size}")
            
            # Convert to OpenCV format
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            print(f"OpenCV image shape: {cv_img.shape}")
            
        print("✅ Screen capture functionality working!")
        
    except Exception as e:
        print(f"❌ Screen capture failed: {e}")

def image_processing_example():
    """Example of image processing with OpenCV and PIL."""
    print("\n🖼️  Image Processing Example")
    print("-" * 30)
    
    try:
        # Create a simple test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[25:75, 25:75] = [0, 255, 0]  # Green square
        
        # OpenCV operations
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        
        # Convert to PIL
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        resized = pil_img.resize((50, 50))
        
        print(f"Original image: {img.shape}")
        print(f"Gray image: {gray.shape}")
        print(f"Blurred image: {blurred.shape}")
        print(f"PIL resized: {resized.size}")
        print("✅ Image processing working!")
        
    except Exception as e:
        print(f"❌ Image processing failed: {e}")

def environment_example():
    """Example of environment variable handling."""
    print("\n🔧 Environment Variables Example")
    print("-" * 30)
    
    try:
        # Load environment variables (if .env file exists)
        load_dotenv()
        
        # Check for common environment variables
        example_vars = ['HOME', 'PATH', 'USER', 'GOOGLE_API_KEY']
        
        for var in example_vars:
            value = os.getenv(var)
            if value:
                if var == 'GOOGLE_API_KEY':
                    # Don't print the actual API key
                    print(f"{var}: {'*' * 20} (hidden)")
                elif len(value) > 50:
                    print(f"{var}: {value[:30]}... (truncated)")
                else:
                    print(f"{var}: {value}")
            else:
                print(f"{var}: Not set")
        
        print("✅ Environment handling working!")
        
    except Exception as e:
        print(f"❌ Environment handling failed: {e}")

def main():
    """Run all examples."""
    print("Wilson Project - uv Usage Examples")
    print("=" * 50)
    
    screen_capture_example()
    image_processing_example()
    environment_example()
    
    print(f"\n🎉 All examples completed!")
    print("\nTo run specific features:")
    print("  • For audio features: uv sync --extra audio")
    print("  • For hardware features: uv sync --extra hardware")
    print("  • For all features: uv sync --extra full")

if __name__ == "__main__":
    main()