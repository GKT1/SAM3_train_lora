import requests
import time

URL = "http://127.0.0.1:8000/v1/models"

print(f"Testing connection to {URL}...")

try:
    start = time.time()
    response = requests.get(URL, headers={"Authorization": "Bearer EMPTY"})
    end = time.time()
    print(f"Response status code: {response.status_code}")
    print(f"Response content: {response.json()}")
    print(f"Time taken: {end - start:.2f} seconds")
    if response.status_code == 200:
        print("✅ SUCCESS: Server is reachable and responding.")
    else:
        print("❌ FAILURE: Server responded with an error.")
except Exception as e:
    print(f"❌ CONNECTION ERROR: {e}")
    print("\nPossible reasons:")
    print("1. Server is not running.")
    print("2. Server is running but still initializing (loading weights).")
    print("3. Server is running on a different port.")
    print("4. Firewall or proxy issues.")
