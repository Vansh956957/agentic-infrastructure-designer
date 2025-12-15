
import { AWSCategory } from './types';
import { ModelIcon, TrainingIcon, DataIcon, EvalIcon } from './components/Icons';

// --- Colors ---
const COLORS = {
    cnn: 'bg-blue-100 border-blue-600 shadow-blue-500/20 dark:bg-blue-900/40 dark:border-blue-500',
    training: 'bg-green-100 border-green-600 shadow-green-500/20 dark:bg-green-900/40 dark:border-green-500',
    aug: 'bg-purple-100 border-purple-600 shadow-purple-500/20 dark:bg-purple-900/40 dark:border-purple-500',
    transfer: 'bg-orange-100 border-orange-600 shadow-orange-500/20 dark:bg-orange-900/40 dark:border-orange-500',
    detection: 'bg-red-100 border-red-600 shadow-red-500/20 dark:bg-red-900/40 dark:border-red-500',
    transformer: 'bg-yellow-100 border-yellow-600 shadow-yellow-500/20 dark:bg-yellow-900/40 dark:border-yellow-500',
    autoencoder: 'bg-teal-100 border-teal-600 shadow-teal-500/20 dark:bg-teal-900/40 dark:border-teal-500',
    gan: 'bg-pink-100 border-pink-600 shadow-pink-500/20 dark:bg-pink-900/40 dark:border-pink-500',
};

// --- Code Templates ---

const LENET_CODE = `import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(16, 120, kernel_size=5),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

# Initialize
model = LeNet5(num_classes=10)
print(f"LeNet-5 Parameters: {sum(p.numel() for p in model.parameters())}")`;

const LENET_TF_CODE = `import tensorflow as tf
from tensorflow.keras import layers, models

def LeNet5(num_classes=10):
    model = models.Sequential()
    # Feature Extractor
    model.add(layers.Conv2D(6, 5, activation='tanh', input_shape=(32, 32, 1)))
    model.add(layers.AveragePooling2D(2))
    model.add(layers.Conv2D(16, 5, activation='tanh'))
    model.add(layers.AveragePooling2D(2))
    model.add(layers.Conv2D(120, 5, activation='tanh'))
    
    # Classifier
    model.add(layers.Flatten())
    model.add(layers.Dense(84, activation='tanh'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

model = LeNet5(num_classes=10)
model.summary()`;

const RESNET_CODE = `import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# Option 1: Load Pretrained
# weights = ResNet18_Weights.DEFAULT
# model = resnet18(weights=weights)

# Option 2: Custom BasicBlock Implementation (Simplified)
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

# Usage
# model = ResNet18() # (Full implementation requires ResNet class)
print("ResNet template loaded.")`;

const RESNET_TF_CODE = `import tensorflow as tf
from tensorflow.keras.applications import ResNet50

# Option 1: Load Pretrained Keras Application
model = ResNet50(weights='imagenet', include_top=True)

# Option 2: Custom Residual Block (Functional API)
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    
    # First Conv
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # Second Conv
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Shortcut path
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x

print("ResNet TF template loaded.")`;

const TRAINING_LOOP_CODE = `import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = MyModel().to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.CrossEntropyLoss()
`;

const TRAINING_LOOP_TF_CODE = `import tensorflow as tf

# 1. Standard Keras Fit (High Level)
# model.compile(
#     optimizer='adam',
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=['accuracy']
# )
# history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# 2. Custom Training Loop (GradientTape)
def train_step(model, images, labels, loss_object, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Usage in loop
# for epoch in range(EPOCHS):
#     for images, labels in train_ds:
#         loss = train_step(model, images, labels, loss_object, optimizer)
`;

const AUGMENTATION_CODE = `from torchvision import transforms

# Training Transforms
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation/Test Transforms (No augmentation, just resize/norm)
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
`;

const AUGMENTATION_TF_CODE = `import tensorflow as tf
from tensorflow.keras import layers

# Keras Preprocessing Layers
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.1),
  layers.RandomContrast(0.1),
])

# Usage inside a model
# inputs = tf.keras.Input(shape=(180, 180, 3))
# x = data_augmentation(inputs)
# x = layers.Rescaling(1./255)(x)
# ...
`;

const VIT_CODE = `import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, Embed_Dim, H/P, W/P) -> (B, Embed_Dim, N_Patches) -> (B, N_Patches, Embed_Dim)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.embedding = PatchEmbedding()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.embedding.n_patches, 768))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=12, dim_feedforward=3072, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)
        self.mlp_head = nn.Linear(768, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.embedding(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        x = self.encoder(x)
        
        cls_output = x[:, 0]
        return self.mlp_head(cls_output)

# model = VisionTransformer()
`;

const VIT_TF_CODE = `import tensorflow as tf
from tensorflow.keras import layers

class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size=16, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size)
        self.flatten = layers.Reshape((-1, embed_dim))

    def call(self, x):
        x = self.proj(x)
        return self.flatten(x)

# Transformer Encoder Block
def transformer_encoder(x, embed_dim, num_heads):
    # Attention
    x1 = layers.LayerNormalization(epsilon=1e-6)(x)
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x1, x1)
    x2 = layers.Add()([attention_output, x])
    
    # MLP
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    x3 = layers.Dense(embed_dim * 4, activation=tf.nn.gelu)(x3)
    x3 = layers.Dense(embed_dim)(x3)
    return layers.Add()([x3, x2])
`;

const GAN_CODE = `import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.gen(z)

class Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)
`;

// --- Categories ---

export const DL_CATEGORIES: AWSCategory[] = [
    {
        name: 'CNN Architectures',
        services: [
            { id: 'cnn-lenet', name: 'LeNet-5', icon: ModelIcon, defaultData: { label: 'LeNet-5', customTheme: COLORS.cnn, code: LENET_CODE, codeTF: LENET_TF_CODE, description: 'Classic CNN for digit recognition.', hyperparams: { num_classes: 10 } } },
            { id: 'cnn-resnet', name: 'ResNet-18/34/50', icon: ModelIcon, defaultData: { label: 'ResNet', customTheme: COLORS.cnn, code: RESNET_CODE, codeTF: RESNET_TF_CODE, description: 'Deep residual networks with skip connections.', hyperparams: { depth: 18, pretrained: 1 } } },
            { id: 'cnn-vgg', name: 'VGG-16', icon: ModelIcon, defaultData: { label: 'VGG-16', customTheme: COLORS.cnn, code: '# VGG PyTorch Placeholder', codeTF: '# VGG Keras Placeholder', description: 'Deep CNN with small 3x3 filters.', hyperparams: { batch_norm: 1 } } },
            { id: 'cnn-efficientnet', name: 'EfficientNet', icon: ModelIcon, defaultData: { label: 'EfficientNet', customTheme: COLORS.cnn, code: '# EfficientNet PyTorch', codeTF: '# EfficientNet Keras', description: 'Compound scaling of depth, width, and resolution.', hyperparams: { version: 'b0' } } },
        ]
    },
    {
        name: 'Training Templates',
        services: [
            { id: 'train-basic', name: 'Basic Training Loop', icon: TrainingIcon, defaultData: { label: 'Training Loop', customTheme: COLORS.training, code: TRAINING_LOOP_CODE, codeTF: TRAINING_LOOP_TF_CODE, description: 'Standard Training boilerlate.', hyperparams: { epochs: 10, lr: 0.001 } } },
            { id: 'train-earlystop', name: 'Early Stopping', icon: TrainingIcon, defaultData: { label: 'Early Stopping', customTheme: COLORS.training, code: '# Early Stopping Class', description: 'Stop training when validation loss plateaus.', hyperparams: { patience: 5 } } },
        ]
    },
    {
        name: 'Preprocessing',
        services: [
            { id: 'aug-transforms', name: 'Transform Pipeline', icon: DataIcon, defaultData: { label: 'Transforms', customTheme: COLORS.aug, code: AUGMENTATION_CODE, codeTF: AUGMENTATION_TF_CODE, description: 'Image augmentation pipelines.', hyperparams: { crop_size: 224 } } },
        ]
    },
    {
        name: 'Transfer Learning',
        services: [
            { id: 'tl-finetune', name: 'Fine-Tuning', icon: TrainingIcon, defaultData: { label: 'Fine-Tune', customTheme: COLORS.transfer, code: '# Fine-Tuning Code\nmodel.fc = nn.Linear(num_ftrs, 2)', description: 'Replace head and retrain.', hyperparams: { freeze_backbone: 1 } } },
        ]
    },
    {
        name: 'Detection & Segmentation',
        services: [
            { id: 'det-yolo', name: 'YOLOv8', icon: ModelIcon, defaultData: { label: 'YOLOv8', customTheme: COLORS.detection, code: '# Ultralytics YOLOv8\nfrom ultralytics import YOLO\nmodel = YOLO("yolov8n.pt")', description: 'Real-time object detection.', hyperparams: { version: 'nano' } } },
            { id: 'seg-unet', name: 'U-Net', icon: ModelIcon, defaultData: { label: 'U-Net', customTheme: COLORS.detection, code: '# U-Net Implementation', description: 'Biomedical image segmentation.', hyperparams: { in_channels: 3 } } },
        ]
    },
    {
        name: 'Transformers (ViT)',
        services: [
            { id: 'vit-base', name: 'Vision Transformer', icon: ModelIcon, defaultData: { label: 'ViT', customTheme: COLORS.transformer, code: VIT_CODE, codeTF: VIT_TF_CODE, description: 'Transformer encoder for image classification.', hyperparams: { patch_size: 16 } } },
        ]
    },
    {
        name: 'Generative (GANs)',
        services: [
            { id: 'gan-dcgan', name: 'DCGAN', icon: ModelIcon, defaultData: { label: 'DCGAN', customTheme: COLORS.gan, code: GAN_CODE, description: 'Deep Convolutional GAN.', hyperparams: { z_dim: 100 } } },
        ]
    }
];
