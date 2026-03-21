#!/bin/bash
# Prussian Dictionary - Start Script

PROJECT_ROOT="$(dirname "$0")"
cd "$PROJECT_ROOT"

echo "🚀 Starting Prussian Dictionary Server..."
echo ""

# Check if venv exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install -q -r requirements.txt
else
    source venv/bin/activate
fi

echo "   Backend API: http://localhost:5000/api/"
echo "   Web UI:      http://localhost:5000/"
echo ""
echo "Press Ctrl+C to stop"
echo ""

cd api
python app.py
