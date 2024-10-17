import torch
import time
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader, ConcatDataset

# Define batch size and transformation for resizing the dataset images
batch_size = 32
resize_tensor = Compose([
    Resize([32, 32]),  # Resize images to 32x32
    ToTensor()  # Convert to torch.Tensor
])

# Load the GTSRB dataset
test_data = datasets.GTSRB(
    root="data",
    split="test",
    download=True,
    transform=resize_tensor
)

# Create a DataLoader for the test dataset (extend to 10,000 images by repeating)
num_repeats = (100000 + len(test_data) - 1) // len(test_data)  # Calculate the number of times to repeat the dataset
extended_test_data = ConcatDataset([test_data] * num_repeats)  # Repeat the dataset
extended_test_loader = DataLoader(extended_test_data, batch_size=batch_size, shuffle=False)

# Load your trained PyTorch model
from CNV_light import cnv  # light version of the CNV
model = cnv(3, 2, 2, 8, 43)  # Replace with your actual architecture
model.load_state_dict(torch.load("/home/ba/PycharmProjects/Brevitas_Thesis/Brevitas_Training/CNV_light_GTSRB/CNV_light_GTSRB_W4A4/model_CNV_light_GTSRB_W4A4.pth"))

# Warm-up function
def warm_up(model, dataloader, device, warmup_iters=1000):
    model.to(device)
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if i >= warmup_iters:
                break  # Run only the specified number of warm-up iterations
            images = images.to(device)
            _ = model(images)

# Function to measure FPS performance
def measure_fps(model, dataloader, device):
    model.to(device)
    model.eval()  # Set model to evaluation mode
    total_images = 0
    start_time = time.time()

    with torch.no_grad():  # Disable gradient calculation for inference
        for images, labels in dataloader:
            images = images.to(device)  # Move data to device (CPU/GPU)
            # Perform inference
            outputs = model(images)
            total_images += len(images)

    total_time = time.time() - start_time
    fps = total_images / total_time
    return fps

# Warm-up and Measure FPS on CPU
device = torch.device("cpu")
print("Warming up CPU...")
warm_up(model, extended_test_loader, device)
cpu_fps = measure_fps(model, extended_test_loader, device)
print(f"FPS on CPU: {cpu_fps:.2f}")

# If GPU is available, warm-up and measure FPS on GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Warming up GPU...")
    warm_up(model, extended_test_loader, device)
    gpu_fps = measure_fps(model, extended_test_loader, device)
    print(f"FPS on GPU: {gpu_fps:.2f}")
else:
    print("GPU is not available.")
