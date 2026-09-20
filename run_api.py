"""
Convenience server script to run the Smart Energy Consumption Prediction System API using Uvicorn.
Usage:
    python run_api.py
"""

import uvicorn

if __name__ == "__main__":
    print("Starting Smart Energy Consumption Prediction System API server...")
    print("Swagger UI documentation available at: http://127.0.0.1:8000/docs")
    print("ReDoc documentation available at: http://127.0.0.1:8000/redoc")
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
