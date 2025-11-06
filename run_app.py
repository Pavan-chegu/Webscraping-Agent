import os
import subprocess
import sys

def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the project root to Python path
    sys.path.insert(0, script_dir)
    
    # Set environment variable for child processes
    env = os.environ.copy()
    env['PYTHONPATH'] = script_dir + os.pathsep + env.get('PYTHONPATH', '')
    
    # Construct the path to the Streamlit app
    app_path = os.path.join(script_dir, "src", "ui", "streamlit_app.py")
    
    # Check if the app file exists
    if not os.path.exists(app_path):
        print(f"Error: Streamlit app not found at {app_path}")
        return 1
    
    # Run the Streamlit app
    print("Starting Web Content RAG Agent...")
    try:
        subprocess.run(["streamlit", "run", app_path], check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())