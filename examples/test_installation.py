#!/usr/bin/env python3
"""
Test script to verify Wilson project dependencies are working correctly.
"""

import sys
import importlib

def test_import(module_name, description=""):
    """Test if a module can be imported successfully."""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} - {description}")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - {description} - Error: {e}")
        return False

def main():
    """Run the import tests."""
    print("Testing Wilson project dependencies...")
    print("=" * 50)
    
    # Core dependencies
    tests = [
        ("cv2", "OpenCV - Computer Vision"),
        ("PIL", "Pillow - Image Processing"),
        ("numpy", "NumPy - Numerical Computing"),
        ("mss", "MSS - Screen Capture"),
        ("dotenv", "Python-dotenv - Environment Variables"),
        ("yaml", "PyYAML - YAML Processing"),
        ("serial", "PySerial - Serial Communication"),
        ("transforms3d", "Transforms3D - 3D Transformations"),
        ("mcp", "MCP - Model Context Protocol"),
        ("google.genai", "Google GenAI - Gemini AI"),
        ("google.generativeai", "Google Generative AI"),
    ]
    
    # Optional dependencies (may fail and that's ok)
    optional_tests = [
        ("taskgroup", "TaskGroup - Python < 3.11 compatibility"),
        ("exceptiongroup", "ExceptionGroup - Python < 3.11 compatibility"),
    ]
    
    success_count = 0
    total_count = len(tests)
    
    for module, description in tests:
        if test_import(module, description):
            success_count += 1
    
    print("\nOptional dependencies (may not be installed):")
    print("-" * 30)
    for module, description in optional_tests:
        test_import(module, description)
    
    print(f"\nResults: {success_count}/{total_count} core dependencies working")
    
    if success_count == total_count:
        print("🎉 All core dependencies are working correctly!")
        return 0
    else:
        print("⚠️  Some dependencies failed to import.")
        return 1

if __name__ == "__main__":
    sys.exit(main())