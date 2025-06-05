#!/usr/bin/env python3
"""
Setup Script for Enhanced Jupyter Notebook Styling
Applies custom CSS for the Sensor Fusion Course notebooks
"""

import os
import shutil
from pathlib import Path

def setup_jupyter_styling():
    """Set up custom CSS for enhanced notebook appearance"""
    
    print("🎨 Setting up enhanced Jupyter notebook styling...")
    
    # Find Jupyter data directory
    home_dir = Path.home()
    jupyter_dir = home_dir / '.jupyter'
    custom_dir = jupyter_dir / 'custom'
    
    # Create directories if they don't exist
    custom_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy custom CSS
    source_css = Path('jupyter_config/custom.css')
    target_css = custom_dir / 'custom.css'
    
    if source_css.exists():
        shutil.copy2(source_css, target_css)
        print(f"✅ Custom CSS installed to: {target_css}")
    else:
        print(f"❌ Source CSS not found: {source_css}")
        return False
    
    # Create a custom.js file for additional enhancements
    custom_js_content = """
// Enhanced Jupyter Notebook JavaScript
// Auto-hide code cells option and other enhancements

define([
    'base/js/namespace',
    'base/js/events'
], function(Jupyter, events) {
    
    // Add custom toolbar buttons
    var add_toolbar_buttons = function() {
        if (!Jupyter.toolbar) return;
        
        // Add a button to toggle all code cells
        Jupyter.toolbar.add_buttons_group([
            {
                'label': 'Toggle Code Cells',
                'icon': 'fa-code',
                'callback': function() {
                    $('.input').slideToggle();
                }
            }
        ]);
        
        // Add a button to toggle outputs
        Jupyter.toolbar.add_buttons_group([
            {
                'label': 'Toggle Outputs',
                'icon': 'fa-eye',
                'callback': function() {
                    $('.output').slideToggle();
                }
            }
        ]);
    };
    
    // Initialize when notebook is ready
    events.on('app_initialized.NotebookApp', add_toolbar_buttons);
    
    // Auto-save functionality enhancement
    events.on('notebook_saved.Notebook', function() {
        console.log('📝 Notebook saved successfully!');
    });
    
    console.log('🚀 Enhanced Jupyter features loaded!');
});
"""
    
    target_js = custom_dir / 'custom.js'
    with open(target_js, 'w', encoding='utf-8') as f:
        f.write(custom_js_content)
    
    print(f"✅ Custom JavaScript installed to: {target_js}")
    
    # Create a simple nbextensions configuration
    nbext_dir = jupyter_dir / 'nbextensions'
    nbext_dir.mkdir(parents=True, exist_ok=True)
    
    print("✨ Jupyter styling setup complete!")
    print("\n📋 Next steps:")
    print("   1. Restart Jupyter if it's running")
    print("   2. Open any notebook to see the enhanced styling")
    print("   3. Use the new toolbar buttons for better navigation")
    
    return True

def create_launch_script():
    """Create a convenient launch script for the notebooks"""
    
    launch_content = """#!/bin/bash
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
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \\
    --NotebookApp.token='' --NotebookApp.password='' \\
    --notebook-dir=notebooks \\
    --ServerApp.allow_origin='*' \\
    --ServerApp.allow_remote_access=True

echo "✅ Jupyter Lab session ended."
"""
    
    with open('launch_notebooks.sh', 'w', encoding='utf-8') as f:
        f.write(launch_content)
    
    # Make executable
    os.chmod('launch_notebooks.sh', 0o755)
    print("✅ Created launch script: launch_notebooks.sh")

def main():
    """Main setup function"""
    print("🔧 Enhanced Jupyter Notebook Setup for Sensor Fusion Course")
    print("=" * 60)
    
    # Set up styling
    if setup_jupyter_styling():
        print("✅ Styling setup successful!")
    else:
        print("❌ Styling setup failed!")
        return
    
    # Create launch script
    create_launch_script()
    
    print("\n🎉 Setup complete!")
    print("\n🚀 To start learning:")
    print("   Option 1: ./launch_notebooks.sh")
    print("   Option 2: jupyter lab")
    print("   Then open: notebooks/lidar/01_introduction_to_lidar.ipynb")
    print("\n✨ Enjoy your enhanced learning experience!")

if __name__ == "__main__":
    main()