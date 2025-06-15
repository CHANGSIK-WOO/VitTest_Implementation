# 1. library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim #SGD, Adam
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from tqdm.auto import tqdm 


# 2. setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Device : {device}")


# 3. set the seed
# what is random seed? : starting point(just like seed) making random numbers.
# for example, real random can not exist in mathmatics perfectly.
# so, Computer always make pseudo-random based on consistent rules.
# if in same seed, output will be always same.
# that means, same seed number can enable experiment result to be reproductable and reliable.
torch.manual_seed(42) #PyTorch CPU based-calculation (ex) torch.rand(3)
torch.cuda.manual_seed(42) #PyTorch GPU based-calculation (ex) Dropout, Sampling 
random.seed(42) # random module (ex) random.randint(0, 10)


# 4. Setting Hyperparameters
BATCH_SIZE = 128 # are optimized in multiple of 8.
EPOCHS = 10
LEARNING_RATE = 3e-4 # 10**-3
PATCH_SIZE = 4
NUM_CLASSES = 10
IMAGE_SIZE = 32
IN_CHANNELS = 3
EMBED_DIM = 256
NUM_HEADS = 8
DEPTH = 6
MLP_DIM = 512
DROP_RATE = 0.1

#CIFAR10 Classes
classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

# 5. Define Image Transformations
transform = transforms.Compose(
    [
        transforms.ToTensor(), # PILImage to Tensor [0,255] --> [0,1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
        # transforms.Normalize(mean, std) : Each Channel Normalzation [0,1] --> [-1,1]
        # = transforms.Normalize((0.5), (0.5)) (ex) (0.5) : float, (0.5,) : tuple
        # from normalization of mean 0.5 & std 0.5, that will help the model to converge faster and to make the numerical computations stable.
    ]
)

# 6. Get datasets
train_datasets = datasets.CIFAR10(root ="data", train=True, download=True, transform=transform)
test_datasets = datasets.CIFAR10(root ="data", train=False, download=True, transform=transform)

print(f"train_datasets count : {len(train_datasets)}, test_datasets count : {len(test_datasets)}")

# 7. Converting our datasets(6) into DataLoaders
# Right now, our data in in the form of Pytorch Datasets.
# DataLoader will turn our data into batches or (mini-batches).
# 1. if we use all data in one hit, will be burdened in memory.
# 2. can be updated per batch in one epoch.

train_dataloader = DataLoader(dataset = train_datasets, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset = test_datasets, batch_size=BATCH_SIZE, shuffle=False)
# test dataloder must not use shuffle for reproductivity of result.

print(f"DataLoader : {train_dataloader, test_dataloader}")
print(f"Length of train_dataloader : {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of test_dataloader : {len(test_dataloader)} batches of {BATCH_SIZE}")


#=====================================================================================#
# 8. Building Vision Transformer from scratch

class PatchEmbedding(nn.Module):
    def __init__(self, img_size : int, patch_size : int, in_channels : int, embed_size : int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_size = embed_size
        # [B, 3, 32, 32] --> [B, 256, H/P, W/P]
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=embed_size, kernel_size=patch_size, stride = patch_size) 
        self.num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(size= [1, 1, embed_size])) 
        # [batch_size, num_patches, embed_size] + + [batch_size, 1, embed_size]  = [batch_size, num_patches + 1, embed_size]
        # cls token is special token which interacts with other patches by self-attention, then 
        # cls token will be abstract of image information, and can be used to classfication.

        self.pos_embed = nn.Parameter(torch.randn(size = [1, self.num_patches + 1, embed_size]))

    def forward(self, x : torch.Tensor):
        B = x.shape[0] # x = [B, C, H, W]
        x = self.proj(x) # [B, 256, H/P, W/P]
        x = x.flatten(2).transpose(1, 2) # [B, num_patches, 256]
        cls_token = self.cls_token.expand(B, -1, -1) # [1, 1, embed_size] --> [B, 1, embed_size]
        x = torch.cat([cls_token, x], dim=1) 
        x = x + self.pos_embed

        return x

class MLP(nn.Module):
    def __init__(self, in_features : int, hidden_features : int, drop_rate : float):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features) # 256 --> 512
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=in_features)
        self.dropout = nn.Dropout(p = drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim : int, num_heads : int, drop_rate : float, qkv_bias : bool = True):
        self.embed_dim = embed_dim # input embedding dimension
        self.num_heads = num_heads # Multi Head Number

        self.head_dim = embed_dim // num_heads # Multi-Head Dimension
        assert self.embed_dim == self.head_dim * self.num_heads, "embed_dim must be divisible by num_heads"

        self.scale = math.sqrt(self.head_dim) # attention score scale
        self.qkv = nn.Linear(in_features=embed_dim, out_features=embed_dim * 3, bias = qkv_bias) # (bs, 256, 768) --> (bs, 256, 768 * 3)
        self.attn_drop = nn.Dropout(p = drop_rate)
        self.projection = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.proj_drop = nn.Dropout(p = drop_rate)

    def forward(self, x):
        bs, num_tokens, embed_dim = x.shape # (bs, patch token 256 + cls token 1=257, 768)
        if embed_dim != self.embed_dim:
            raise ValueError(f"Input Dimension of x {embed_dim} does not equal to the expected dimension {self.dim}")
        
        qkv = self.qkv(x) # (bs, num_tokens + 1, dim *3) +1 means [CLS] token in 1st position of embedding for classification.
        qkv = qkv.reshape(bs, num_tokens, 3, self.num_heads, self.head_dim) # (bs, 257, 768 * 3) --> (bs, 257, 3, 12, 64)
        qkv =  # (bs, 257, 3, 12, 64) --> (3, bs, 12, 257, 64)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = q @ k.tranpose(-2, -1) * self.scale # (3, bs, 12, 257, 64) --> (bs, 12, 257, 257)
        attn_scores = attn_scores.softmax(dim=-1) # (bs, 12, 257, 257) --> (bs, 12, 257, 257)
        attn_scores = self.attn_drop(attn_scores)

        mha = attn_scores @ v # (bs, 12, 257, 257) --> (bs, 12, 257, 64)
        mha = mha.transpose(1, 2).flatten(2) # (bs, 12, 257, 64) --> (bs, 257, 12, 64) --> (bs, 257, 768)

        x = self.projection(mha)
        x = self.proj_drop(x)

        return mha 

    
class VisionTransformerBlock(nn.Module):
    def __init__(self, embed_dim : int, num_heads : int, mlp_dim : int, drop_rate : float):
        super().__init__()
        self.norm1 = nn.LayerNorm(normalized_shape=embed_dim, eps = 1e-6)
        self.attn = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, drop_rate=drop_rate)
        self.norm2 = nn.LayerNorm(normalized_shape=embed_dim, eps = 1e-6)
        self.mlp = MLP(in_features=embed_dim, hidden_features=mlp_dim, drop_rate=drop_rate)

    def forward(self, x):
        x = self.norm1(x)
        x = self.attn(x) + x
        x = self.norm2(x)
        x = self.mlp(x) + x

        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, 
                 img_size : int, 
                 patch_size : int, 
                 in_channels : int, 
                 num_classes : int, 
                 embed_dim : int, 
                 depth : int, 
                 num_heads : int, 
                 mlp_dim : int, 
                 drop_rate : float):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size = img_size, patch_size = patch_size, in_channels = in_channels, embed_size =  embed_dim)
        self.encoder = nn.ModuleList(
            [
            VisionTransformerBlock(embed_dim = embed_dim, num_heads =  num_heads, mlp_dim =  mlp_dim, drop_rate = drop_rate) 
            for _ in range(depth)
            ]
        )
        # ModuleList : can be accessble to each block. can apply different forward each block.
        # Sequential : cant be accessible to each black. just forward.

        self.norm = nn.LayerNorm(normalized_shape =  embed_dim)
        self.head = nn.Linear(in_features = embed_dim, out_features = num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.norm(x)
        cls_token = x[:, 0] # == x[:, 0, :], (bs, embed_dim)
        x = self.head(cls_token) #  (bs, num_classes) 

        return x 
    
#=====================================================================================#
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad() # initialize gradient

        pred = model(images)
        loss = criterion(pred, labels) # CrossEntropyLoss
        
        loss.backward() # calculate gradient from back-propagation --> W.grad = dLoss/dW
        optimizer.step() # update parameter --> W = W - lr * W.grad

        total_loss += loss.item() * images.shape[0]
        correct += (pred.argmax(1) == labels).sum().items()


    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def evaluate(model, loader):
    model.eval() #change dropout, batchnorm condition to Eval Mode
    with torch.no_grad(): # no gradient calculation
        for images, labels in loader: # load batch
            images, labels = images.to(device), labels.to(device)
            
            pred = model(images) # [bs, 10]
            correct += (pred.argmax(dim=1) == labels).sum().item()
    
    return correct / len(loader.dataset)

#=====================================================================================#
model = VisionTransformer(img_size = IMAGE_SIZE, 
                          patch_size = PATCH_SIZE, 
                          in_channels = IN_CHANNELS,
                          num_classes =  NUM_CLASSES,
                          embed_dim =  EMBED_DIM,
                          depth = DEPTH,
                          num_heads =  NUM_HEADS,
                          mlp_dim =  MLP_DIM,
                          drop_rate = DROP_RATE).to(device) #to CUDA

# Loss (Multi-Class Classification Loss : softmax)
criterion = nn.CrossEntropyLoss()

# Ready to update all parameters learnable in model via Adam Optimizer
optimizer = optim.Adam(params = model.parameters(), lr = LEARNING_RATE)

plt.ion()  # Interactive mode ON
train_accuracies, train_losses, test_accuracies = list(), list(), list()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for epoch in tqdm(range(EPOCHS)): #loop 10 times
    train_loss, train_acc = train(model = model, loader = train_dataloader, optimizer = optimizer, criterion = criterion)
    test_acc = evaluate(model = model, loader = test_dataloader)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    train_losses.append(train_loss)

    print(f"Epoch : {epoch} / {EPOCHS}, Train Loss : {train_loss:.4f}, Train Acc. : {100 * train_acc:.1f}%, Test Acc. : {100 * test_acc:.1f}%")
    
    #clear and update plots
    ax1.cla()
    ax2.cla()

    # Accuracy plot
    ax1.plot(train_accuracies, label="Train Acc")
    ax1.plot(test_accuracies, label="Test Acc")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    # Loss plot
    ax2.plot(train_losses, label="Train Loss", color='red')
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.suptitle(f"Epoch {epoch+1}/{EPOCHS}")
    plt.pause(0.1)  

plt.ioff()  # Interactive Mode Off
plt.show()  # Fix last graph


    