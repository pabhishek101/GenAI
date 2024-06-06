##code for human parsing using pretrained model from hugging face ##

from PIL import Image
from transformers import pipeline
import matplotlib.pyplot as plt

def visualize_parsing_results(image_path,color_palette):
    # Load the human parsing model
    parsing_pipeline = pipeline("image-segmentation", model="matei-dorian/segformer-b5-finetuned-human-parsing")
    # Load the image
    image = Image.open(image_path)
    # Perform human parsing
    parsing_results = parsing_pipeline(image)
    
    # Get the size of the first mask (assuming all masks have the same size)
    mask_size = parsing_results[0]["mask"].size

    # Create a new image with the same size and RGB mode
    parsed_image = Image.new("RGB", mask_size)

    # Iterate over the parsing results
    for result in parsing_results:
        label = result["label"]
        mask = result["mask"]
        # Get the color for the label from the palette, or black if not found
        color = color_palette.get(label, (0, 0, 0))
        # Paste the colored mask onto the parsed image
        parsed_image.paste(color, mask=mask)
    plt.imshow(parsed_image)
    plt.show()
    
    return parsed_image



color_palette = {
        'Background': (0, 0, 0),      # Background
        'Hat': (255, 0, 0),    # Hat
        'Hair': (128, 0, 0),    # Hair
        'Glove': (255, 255, 0),  # Glove
        'Sunglasses': (128, 128, 0),  # Sunglasses
        'Upper-clothes': (0, 255, 0),    # UpperClothes
        'Dress': (128, 255, 0),  # Dress
        'Coat': (0, 128, 0),    # Coat
        'Socks': (0, 255, 255),  # Socks
        'Pants': (0, 128, 128),  # Pants
        'Jumpsuits': (0, 0, 255),   # Jumpsuits
        'Scarf': (255, 0, 255), # Scarf
        'Skirt': (128, 0, 128), # Skirt
        'Face': (255, 255, 255), # Face
        'Left-arm': (192, 192, 192), # Left-arm
        'Right-arm': (128, 128, 128), # Right-arm
        'Left-leg': (64, 64, 64), # Left-leg
        'Right-leg': (0, 64, 64), # Right-leg
        'Left-shoe': (128, 0, 64), # Left-shoe
        'Right-shoe': (64, 0, 32), # Right-shoe
        'Belt': (0,125,74)
    }

# Create and display the color-coded parsed image
parsed_image = visualize_parsing_results('/content/output_06.png', color_palette)

