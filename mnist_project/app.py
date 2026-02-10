import torch
from flask import Flask, render_template, request
from PIL import Image, ImageOps
from torchvision import transforms
from model import SimpleCNN

app = Flask(__name__)

# Load Model
model = SimpleCNN()
# Load trained weights
model.load_state_dict(torch.load("mnist_cnn.pth"))
model.eval()  # Set to evaluation mode (disables dropout/batchnorm updates)

def transform_image(image_bytes):
    """Prepares user image to match MNIST format."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    image = Image.open(image_bytes)
    # Invert image (MNIST is white text on black background; users usually upload black on white)
    image = ImageOps.invert(image.convert('RGB')) 
    return transform(image).unsqueeze(0) # Add batch dimension (1, 1, 28, 28)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded", 400
        
        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400

        try:
            tensor = transform_image(file)
            with torch.no_grad(): # Disable gradients for faster inference
                output = model(tensor)
                prediction = output.argmax(dim=1).item()
        except Exception as e:
            print(f"Error: {e}")

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)