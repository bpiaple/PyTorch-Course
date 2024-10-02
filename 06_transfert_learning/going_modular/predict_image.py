# Try to make a prediction on the custom image
from typing import List, Tuple
from PIL import Image

import torch
from torch import nn
from torchvision.transforms import v2 as transforms

def predict_image(model: nn.Module,
                    image_path: str,
                    transform: transforms.Compose,
                    class_names: List[str]) -> Tuple[str, float]:
        """Predicts the class of an image.
    
        Args:
            model (nn.Module): A PyTorch neural network model.
            image_path (str): The path to the image.
            transform (transforms.Compose): A PyTorch transformation.
            class_names (List[str]): A list of class names.

        Returns:
            Tuple[str, float]: A tuple containing the class name and the probability.
        """
        # Load the image
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        image = image.unsqueeze(0)

        # Make a prediction
        model.eval()
        with torch.inference_mode():
            predictions = model(image)
            predicted_probabilities = torch.softmax(predictions, dim=1)
            predicted_class_index = torch.argmax(predicted_probabilities, dim=1)
            predicted_class_name = class_names[predicted_class_index]
            predicted_probability = predicted_probabilities[0][predicted_class_index].item()

        return predicted_class_name, predicted_probability