import tkinter as tk
import torch
import logging
from lime import lime_image
from visualization import EnhancedLIMEGradCAMVisualization
from models import GradCAM
from config import CONFIG
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(CONFIG['RESULTS_DIR'], 'app.log')),
        logging.StreamHandler()
    ]
)

def load_model():
    """
    Load the pre-trained model for oral cancer detection.
    Replace this with your actual model loading logic.
    """
    try:
        # Placeholder: Load a PyTorch model (e.g., ResNet, custom CNN)
        # Example: model = torchvision.models.resnet18(pretrained=False)
        # model.load_state_dict(torch.load('path_to_model.pth'))
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        model.eval()
        model.to(CONFIG['DEVICE'])
        logging.info("Model loaded successfully")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")
        raise

def load_explainer():
    """
    Initialize the LIME explainer for image explanations.
    """
    try:
        explainer = lime_image.LimeImageExplainer()
        logging.info("LIME explainer initialized")
        return explainer
    except Exception as e:
        logging.error(f"Failed to initialize LIME explainer: {str(e)}")
        raise

def main():
    """
    Main function to initialize and run the visualization application.
    """
    # Initialize Tkinter root
    root = tk.Tk()

    # Load model and explainer
    try:
        model = load_model()
        explainer = load_explainer()
    except Exception as e:
        tk.messagebox.showerror("Error", f"Initialization failed: {str(e)}")
        root.destroy()
        return

    # Create the visualization application
    try:
        app = EnhancedLIMEGradCAMVisualization(root, model, explainer)
        logging.info("Visualization application initialized")
    except Exception as e:
        logging.error(f"Failed to initialize visualization: {str(e)}")
        tk.messagebox.showerror("Error", f"Failed to initialize visualization: {str(e)}")
        root.destroy()
        return

    # Start the Tkinter main loop
    try:
        root.mainloop()
        logging.info("Application started")
    except Exception as e:
        logging.error(f"Application crashed: {str(e)}")
        tk.messagebox.showerror("Error", f"Application crashed: {str(e)}")
        root.destroy()

if __name__ == "__main__":
    main()
