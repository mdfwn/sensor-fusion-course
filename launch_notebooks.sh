#!/bin/bash
# Enhanced Jupyter Lab Launch Script for Sensor Fusion Course

echo "🚀 Starting Enhanced Jupyter Lab for Sensor Fusion Course..."
echo "📚 Features enabled:"
echo "   • Custom styling and themes"
echo "   • Enhanced code highlighting"
echo "   • Interactive visualizations"
echo "   • Progress tracking"
echo ""

# Set environment variables for better display
export JUPYTER_ENABLE_LAB=yes
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Launch Jupyter Lab with custom configuration
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
    --NotebookApp.token='' --NotebookApp.password='' \
    --notebook-dir=notebooks \
    --ServerApp.allow_origin='*' \
    --ServerApp.allow_remote_access=True

echo "✅ Jupyter Lab session ended."
