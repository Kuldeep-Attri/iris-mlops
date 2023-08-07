#!/bin/bash

# Run the below command to start backend server
uvicorn src.serve:app --host 0.0.0.0 --port 8000