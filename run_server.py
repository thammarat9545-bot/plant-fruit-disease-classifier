"""
Convenient launcher script for the Plant & Fruit Disease Classifier Web Server.
Automatically starts Uvicorn and opens the browser.
"""

import sys
import webbrowser
import threading
import time
import uvicorn

def open_browser():
    time.sleep(1.5)
    url = "http://127.0.0.1:8000"
    print(f"\n[System] Opening application in browser: {url}")
    webbrowser.open(url)

if __name__ == "__main__":
    print("=" * 65)
    print(" 🌿 Plant & Fruit Disease AI Classifier Web Server")
    print("   - API & Frontend: http://127.0.0.1:8000")
    print("   - Swagger Docs:   http://127.0.0.1:8000/docs")
    print("=" * 65)

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("app.server:app", host="127.0.0.1", port=8000, reload=True)
