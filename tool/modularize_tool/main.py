import logging
import os
from config import CONFIG
from gui import EnhancedLIMEGradCAMVisualization
import tkinter as tk

def setup_logging():
    """Configure logging for the application."""
    log_dir = os.path.join(CONFIG['RESULTS_DIR'], 'debug')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'debug.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("Logging initialized for oral cancer detection application")

def main():
    """Main entry point for the oral cancer detection application."""
    setup_logging()
    logging.info("Starting oral cancer detection application")
    
    try:
        # Initialize Tkinter root
        root = tk.Tk()
        # Create GUI instance
        app = EnhancedLIMEGradCAMVisualization(root)
        # Start the Tkinter main loop
        root.mainloop()
        logging.info("Application closed")
    except Exception as e:
        logging.error(f"Application failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
