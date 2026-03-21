#!/bin/bash
# Prussian Dictionary - Start Script

cd "$(dirname "$0")/api"

echo "🚀 Starting Prussian Dictionary Server..."
echo ""
echo "   Backend API: http://localhost:5000/api/"
echo "   Web UI:      http://localhost:5000/"
echo ""
echo "Press Ctrl+C to stop"
echo ""

source ../v2/venv/bin/activate
python app.py
