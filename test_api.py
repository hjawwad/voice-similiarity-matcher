#!/usr/bin/env python3
"""
Test script for the Voice Matching Flask API
"""

import requests
import json

def test_api():
    """Test the voice matching API with sample requests."""
    
    # API endpoint
    url = "http://localhost:5000/compare_voices"
    
    # Test files (you'll need to provide actual audio files)
    # For testing, you can use any two audio files
    audio1_path = "sample1.wav"  # Replace with actual audio file path
    audio2_path = "sample2.wav"  # Replace with actual audio file path
    
    try:
        # Prepare files for upload
        files = {
            'audio1': open(audio1_path, 'rb'),
            'audio2': open(audio2_path, 'rb')
        }
        
        print("Sending request to voice matching API...")
        print(f"Audio 1: {audio1_path}")
        print(f"Audio 2: {audio2_path}")
        
        # Send POST request
        response = requests.post(url, files=files)
        
        # Close file handles
        files['audio1'].close()
        files['audio2'].close()
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            print("\n✅ API Response:")
            print(json.dumps(result, indent=2))
            
            print(f"\n📊 Summary:")
            print(f"Similarity Score: {result['similarity_score']:.4f}")
            print(f"Same Person: {result['is_same_person']}")
            print(f"Conclusion: {result['conclusion']}")
            print(f"Processing Time: {result['execution_time_seconds']} seconds")
            print(f"Memory Usage: {result['memory_usage_mb']} MB")
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Please provide valid audio file paths in the script")
    except requests.exceptions.ConnectionError:
        print("❌ Connection error: Make sure the Flask app is running on localhost:5000")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_health():
    """Test the health check endpoint."""
    try:
        response = requests.get("http://localhost:5000/health")
        if response.status_code == 200:
            print("✅ Health check passed:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure Flask app is running.")

def test_documentation():
    """Test the documentation endpoint."""
    try:
        response = requests.get("http://localhost:5000/")
        if response.status_code == 200:
            print("✅ API Documentation:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ Documentation endpoint failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure Flask app is running.")

if __name__ == "__main__":
    print("🧪 Testing Voice Matching API")
    print("=" * 50)
    
    print("\n1. Testing health endpoint...")
    test_health()
    
    print("\n2. Testing documentation endpoint...")
    test_documentation()
    
    print("\n3. Testing voice comparison endpoint...")
    print("Note: Update audio file paths in the script before running this test")
    # Uncomment the line below after updating the audio file paths
    # test_api()

