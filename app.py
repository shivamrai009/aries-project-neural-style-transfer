import torch
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# Load images
def load_image(image_path, transform=None, max_size=400, shape=None):
    image = Image.open(image_path)
    if max_size:
        size = max_size if max(image.size) > max_size else max(image.size)
        if shape:
            size = shape
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), 
                                 (0.229, 0.224, 0.225))])
    image = transform(image).unsqueeze(0)
    return image

# VGG19 model
class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.features = models.vgg19(pretrained=True).features[:21].eval()
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.features(x)

# Load content and style images
content_img = load_image('path_to_content_image.jpg')
style_img = load_image('path_to_style_image.jpg')

# Instantiate model and optimizer
model = VGG19().to(device)
optimizer = optim.Adam([target_img.requires_grad_()], lr=0.003)

# Define loss functions
def content_loss(content, target):
    return torch.mean((content - target)**2)

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(b * c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram.div(b * c * h * w)

def style_loss(style, target):
    G = gram_matrix(target)
    A = gram_matrix(style)
    return torch.mean((G - A)**2)

# Training loop
epochs = 3000
for epoch in range(epochs):
    target_features = model(target_img)
    content_features = model(content_img)
    style_features = model(style_img)

    c_loss = content_loss(content_features, target_features)
    s_loss = style_loss(style_features, target_features)
    total_loss = c_loss + s_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f'Epoch {epoch}, Total Loss: {total_loss.item()}')

# Display result
plt.imshow(target_img.cpu().squeeze().permute(1, 2, 0).clamp(0, 1).numpy())
plt.show()
