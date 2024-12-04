import torch
import numpy as np
import shap

def explain_model_with_shap(model_path, input_data):
    """
    Explain the model's predictions using SHAP for a saved PyTorch model.
    
    Parameters:
    - model_path (str): Path to the saved PyTorch model.
    - input_data (numpy.ndarray): The input data for which SHAP values are to be computed.
    
    Returns:
    - shap_values (list of numpy.ndarray): SHAP values for the input data.
    """
    
    # Load the saved PyTorch model
    model = torch.load(model_path)
    model.eval()  # Set the model to evaluation mode
    
    # Ensure input_data is a numpy array, if it's not already
    input_data = np.array(input_data)
    
    # Convert input data to a PyTorch tensor
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    
    # Define the prediction function for SHAP
    def predict_fn(inputs):
        inputs = torch.tensor(inputs, dtype=torch.float32)
        outputs = model(inputs)
        return outputs.detach().numpy()  # Return as numpy array for SHAP
    
    # Initialize SHAP DeepExplainer
    explainer = shap.DeepExplainer(predict_fn, input_tensor)
    
    # Compute SHAP values for the input data
    shap_values = explainer.shap_values(input_data)
    
    return shap_values

# Example usage
model_path = "example_model.pth"
input_data = np.random.randn(10, 1, 1024)  # Replace with your actual input data

# Get SHAP values for the input data
shap_values = explain_model_with_shap(model_path, input_data)

# Visualize SHAP values for the first output dimension
shap.summary_plot(shap_values[0], input_data)  # [0] for the first output dimension
