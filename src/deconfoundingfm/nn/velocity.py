import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
# ----------------------------------------------------------------------------- 
# Configuration 
# -----------------------------------------------------------------------------

@dataclass
class FMVelocityConfig:
    dim_y: int
    hidden: int = 64
    layers: int = 2
    context_dim: int = 0     


# ----------------------------------------------------------------------------- 
# Base class 
# -----------------------------------------------------------------------------

class BaseVelocityField(nn.Module):
    is_image: bool = False      # default: vector-valued
    requires_onehot_context: bool = False
    
    def forward(
        self,
        y: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        raise NotImplementedError


# ----------------------------------------------------------------------------- 
# Utility 
# -----------------------------------------------------------------------------

def build_mlp(in_dim: int, out_dim: int, hidden: int, layers: int) -> nn.Sequential:
    layers_list = []
    prev = in_dim
    for _ in range(layers):
        layers_list.append(nn.Linear(prev, hidden))
        layers_list.append(nn.SiLU())
        prev = hidden
    layers_list.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers_list)


# ----------------------------------------------------------------------------- 
# Time + Context velocity field 
# -----------------------------------------------------------------------------

class MLPVelocityField(BaseVelocityField):
    """
    v_theta(y, t, ctx) = MLP( concat(y, [t], [ctx]) )

    - context is optional
    - if context_dim > 0 but context=None, 
      a zero context vector is substituted.

    This makes the field usable in per-arm mode (no context)
    or in conditional mode (shared field with context).
    """

    is_image = False      # default: vector-valued
    requires_onehot_context = False
    
    def __init__(self, cfg: FMVelocityConfig):
        super().__init__()
        self.cfg = cfg

        # y + t + context (always present)
        in_dim = cfg.dim_y + 1 + cfg.context_dim
        self.net = build_mlp(in_dim, cfg.dim_y, cfg.hidden, cfg.layers)

    def forward(self, y, t, context):
        """
        Assumes:
            y:       (B, dim_y)
            t:       (B,) or (B,1)
            context: (B, context_dim)  ALWAYS provided
        """
        
        # Ensure t is (B,1) without branching
        B = y.shape[0]
        t = t.view(B, 1)

        # Concatenate fixed inputs (no conditional cat)
        x = torch.cat([y, t, context], dim=-1)
        return self.net(x)

# ----------------------------------------------------------------------------- 
# UNET velocity field from https://github.com/Michedev/flow-matching-mnist/blob/main/models/velocity_architectures/unet_class_cond.py
# -----------------------------------------------------------------------------        

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.embedding_l = nn.Linear(3, dim)

    def forward(self, time):
        device = time.device
        time = torch.cat([time, torch.cos(time), torch.sin(time)], dim=-1)
        return self.embedding_l(time)

class ClassNorm(nn.Module):

    def __init__(self, num_classes: int, hidden_dim, out_dim: int):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.beta_nn = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.gamma_nn = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, y):
        beta = self.beta_nn(y)
        gamma = self.gamma_nn(y)
        beta = beta.view(beta.shape[0], beta.shape[1], 1, 1)
        gamma = gamma.view(gamma.shape[0], gamma.shape[1], 1, 1)
        return (1 + gamma) * x + beta


class DoubleConvClassCond(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation1 = nn.SiLU(inplace=True)
        self.norm1 = ClassNorm(num_classes=num_classes, hidden_dim=num_classes, out_dim=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation2 = nn.SiLU(inplace=True)
        self.norm2 = ClassNorm(num_classes=num_classes, hidden_dim=num_classes, out_dim=out_channels)

    def forward(self, x, y):
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.norm1(x, y)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.norm2(x, y)
        return x

class UNet(nn.Module):
    is_image = True      
    requires_onehot_context = True
    
    def __init__(self, in_channels, out_channels, num_classes: int, c=64):
        super().__init__()
        
        # Time embeddings for each skip connection
        self.time_mlp1 = nn.Sequential(
            SinusoidalPositionEmbeddings(c),
            nn.Linear(c, c),
            nn.SiLU(),
            nn.Linear(c, c)
        )
        
        self.time_mlp2 = nn.Sequential(
            SinusoidalPositionEmbeddings(c),
            nn.Linear(c, 2*c),
            nn.SiLU(),
            nn.Linear(2*c, 2*c)
        )
        
        self.time_mlp3 = nn.Sequential(
            SinusoidalPositionEmbeddings(c),
            nn.Linear(c, 4*c),
            nn.SiLU(),
            nn.Linear(4*c, 4*c)
        )
        
        # Encoder
        self.conv1 = DoubleConvClassCond(in_channels, c, num_classes)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConvClassCond(c, 2*c, num_classes)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConvClassCond(2*c, 4*c, num_classes)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConvClassCond(4*c, 8*c, num_classes)

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(8*c, 4*c, kernel_size=2, stride=2)
        self.conv5 = DoubleConvClassCond(12*c, 4*c, num_classes)  # 4c + 4c + 4c (skip + up + time)
        self.upconv2 = nn.ConvTranspose2d(4*c, 2*c, kernel_size=2, stride=2)
        self.conv6 = DoubleConvClassCond(6*c, 2*c, num_classes)   # 2c + 2c + 2c
        self.upconv1 = nn.ConvTranspose2d(2*c, c, kernel_size=2, stride=2)
        self.conv7 = DoubleConvClassCond(3*c, c, num_classes)     # c + c + c
        
        self.final_conv = nn.Conv2d(c, out_channels, kernel_size=1)

    
    def forward(self, y, t, context):
        x = y
        y_cls = context
        # Time embeddings
        t = t.view(-1, 1)
        if t.numel() == 1:
            t = t.expand(x.shape[0], -1)
        t1 = self.time_mlp1(t)        # Shape: [batch, c]
        t2 = self.time_mlp2(t)        # Shape: [batch, 2c]
        t3 = self.time_mlp3(t)        # Shape: [batch, 4c]
        
        # Encoder
        conv1 = self.conv1(x, y_cls)              # Shape: [batch, c, H, W]
        pool1 = self.pool1(conv1)             # Shape: [batch, c, H/2, W/2]
        
        conv2 = self.conv2(pool1, y_cls)          # Shape: [batch, 2c, H/2, W/2]
        pool2 = self.pool2(conv2)             # Shape: [batch, 2c, H/4, W/4]
        
        conv3 = self.conv3(pool2, y_cls)          # Shape: [batch, 4c, H/4, W/4]
        pool3 = self.pool3(conv3)             # Shape: [batch, 4c, H/8, W/8]
        
        conv4 = self.conv4(pool3, y_cls)          # Shape: [batch, 8c, H/8, W/8]
        
        # Helper function for padding
        def pad_if_needed(upsampled, skip):
            if (upsampled.shape[-1] + 1) == skip.shape[-1]:
                return F.pad(upsampled, (0, 1, 0, 1), mode='replicate')
            return upsampled
        # Decoder Step 1
        up3 = self.upconv3(conv4)              # Shape: [batch, 4c, H/4, W/4]
        up3 = pad_if_needed(up3, conv3)        # Ensure up3 matches conv3 dimensions
        # Reshape t3 to [batch, 4c, H/4, W/4]
        t_emb3 = t3.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, up3.size(2), up3.size(3))  
        # Alternatively, use expand to save memory:
        # t_emb3 = t3.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, up3.size(2), up3.size(3))
        concat3 = torch.cat([up3, conv3, t_emb3], dim=1)   # Shape: [batch, 12c, H/4, W/4]
        conv5 = self.conv5(concat3, y_cls)         # Shape: [batch, 4c, H/4, W/4]

        # Decoder Step 2
        up2 = self.upconv2(conv5)              # Shape: [batch, 2c, H/2, W/2]
        up2 = pad_if_needed(up2, conv2)        # Ensure up2 matches conv2 dimensions
        # Reshape t2 to [batch, 2c, H/2, W/2]
        t_emb2 = t2.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, up2.size(2), up2.size(3))
        concat2 = torch.cat([up2, conv2, t_emb2], dim=1)   # Shape: [batch, 6c, H/2, W/2]
        conv6 = self.conv6(concat2, y_cls)         # Shape: [batch, 2c, H/2, W/2]

        # Decoder Step 3
        up1 = self.upconv1(conv6)              # Shape: [batch, c, H, W]
        up1 = pad_if_needed(up1, conv1)        # Ensure up1 matches conv1 dimensions
        # Reshape t1 to [batch, c, H, W]
        t_emb1 = t1.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, up1.size(2), up1.size(3))
        concat1 = torch.cat([up1, conv1, t_emb1], dim=1)   # Shape: [batch, 3c, H, W]
        conv7 = self.conv7(concat1, y_cls)         # Shape: [batch, c, H, W]

        # Final Convolution
        return self.final_conv(conv7)          # Shape: [batch, out_channels, H, W]

# ---------------------------------------------------
# Modification of Unet to handle continuous context X
# ---------------------------------------------------

class FiLM(nn.Module):
    def __init__(self, x_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * out_dim),
        )

    def forward(self, h, x):
        """
        h: (B, C, H, W)  feature map
        x: (B, x_dim)    continuous covariate (e.g. theta)
        """
        gamma_beta = self.net(x)               # (B, 2C)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.view(-1, h.shape[1], 1, 1)
        beta  = beta.view(-1, h.shape[1], 1, 1)
        return (1 + gamma) * h + beta


class DoubleConvClassXCond(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, x_dim, film_hidden: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.act1 = nn.SiLU(inplace=True)
        self.norm1 = ClassNorm(num_classes, hidden_dim=num_classes, out_dim=out_channels)
        self.film1 = FiLM(x_dim, hidden_dim=film_hidden, out_dim=out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.act2 = nn.SiLU(inplace=True)
        self.norm2 = ClassNorm(num_classes, hidden_dim=num_classes, out_dim=out_channels)
        self.film2 = FiLM(x_dim, hidden_dim=film_hidden, out_dim=out_channels)

    def forward(self, x, a_onehot, x_cont):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.norm1(x, a_onehot)
        x = self.film1(x, x_cont)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.norm2(x, a_onehot)
        x = self.film2(x, x_cont)

        return x


class UNetX(nn.Module):
    is_image = True
    requires_onehot_context = False  # context is not fully one-hot anymore (only A is)
    
    def __init__(self, in_channels, out_channels, num_classes: int, 
                 x_dim: int, 
                 c=64, 
                 film_hidden: int = 64,
                 film_encoder: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.x_dim = x_dim
        self.film_encoder = film_encoder
        
        # Time embeddings for each skip connection
        self.time_mlp1 = nn.Sequential(
            SinusoidalPositionEmbeddings(c),
            nn.Linear(c, c),
            nn.SiLU(),
            nn.Linear(c, c)
        )
        
        self.time_mlp2 = nn.Sequential(
            SinusoidalPositionEmbeddings(c),
            nn.Linear(c, 2*c),
            nn.SiLU(),
            nn.Linear(2*c, 2*c)
        )
        
        self.time_mlp3 = nn.Sequential(
            SinusoidalPositionEmbeddings(c),
            nn.Linear(c, 4*c),
            nn.SiLU(),
            nn.Linear(4*c, 4*c)
        )
        
        # Encoder
        if self.film_encoder:
            self.conv1 = DoubleConvClassXCond(in_channels, c, num_classes, x_dim, film_hidden=film_hidden)
            self.pool1 = nn.MaxPool2d(2)
            self.conv2 = DoubleConvClassXCond(c, 2*c, num_classes, x_dim, film_hidden=film_hidden)
            self.pool2 = nn.MaxPool2d(2)
            self.conv3 = DoubleConvClassXCond(2*c, 4*c, num_classes, x_dim, film_hidden=film_hidden)
            self.pool3 = nn.MaxPool2d(2)
            self.conv4 = DoubleConvClassXCond(4*c, 8*c, num_classes, x_dim, film_hidden=film_hidden)
        else:
            self.conv1 = DoubleConvClassCond(in_channels, c, num_classes)
            self.pool1 = nn.MaxPool2d(2)
            self.conv2 = DoubleConvClassCond(c, 2*c, num_classes)
            self.pool2 = nn.MaxPool2d(2)
            self.conv3 = DoubleConvClassCond(2*c, 4*c, num_classes)
            self.pool3 = nn.MaxPool2d(2)
            self.conv4 = DoubleConvClassCond(4*c, 8*c, num_classes)

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(8*c, 4*c, kernel_size=2, stride=2)
        self.conv5 = DoubleConvClassXCond(12*c, 4*c, num_classes, x_dim, film_hidden=film_hidden)  # 4c + 4c + 4c (skip + up + time)
        self.upconv2 = nn.ConvTranspose2d(4*c, 2*c, kernel_size=2, stride=2)
        self.conv6 = DoubleConvClassXCond(6*c, 2*c, num_classes, x_dim, film_hidden=film_hidden)   # 2c + 2c + 2c
        self.upconv1 = nn.ConvTranspose2d(2*c, c, kernel_size=2, stride=2)
        self.conv7 = DoubleConvClassXCond(3*c, c, num_classes, x_dim, film_hidden=film_hidden)     # c + c + c
        
        self.final_conv = nn.Conv2d(c, out_channels, kernel_size=1)

    
    def forward(self, y, t, context):
        x = y

        # context is assumed to be concatenated as [X_continuous, A_onehot]
        x_cont = context[:, :self.x_dim]
        a_onehot = context[:, self.x_dim:self.x_dim + self.num_classes]

        # Time embeddings
        t = t.view(-1, 1)
        if t.numel() == 1:
            t = t.expand(x.shape[0], -1)
        t1 = self.time_mlp1(t)        # Shape: [batch, c]
        t2 = self.time_mlp2(t)        # Shape: [batch, 2c]
        t3 = self.time_mlp3(t)        # Shape: [batch, 4c]
        
        # Encoder
        if self.film_encoder:
            conv1 = self.conv1(x, a_onehot, x_cont)
            pool1 = self.pool1(conv1)
        
            conv2 = self.conv2(pool1, a_onehot, x_cont)
            pool2 = self.pool2(conv2)
        
            conv3 = self.conv3(pool2, a_onehot, x_cont)
            pool3 = self.pool3(conv3)
        
            conv4 = self.conv4(pool3, a_onehot, x_cont)
        else:
            conv1 = self.conv1(x, a_onehot)
            pool1 = self.pool1(conv1)
        
            conv2 = self.conv2(pool1, a_onehot)
            pool2 = self.pool2(conv2)
        
            conv3 = self.conv3(pool2, a_onehot)
            pool3 = self.pool3(conv3)
        
            conv4 = self.conv4(pool3, a_onehot)

        
        # Helper function for padding
        def pad_if_needed(upsampled, skip):
            if (upsampled.shape[-1] + 1) == skip.shape[-1]:
                return F.pad(upsampled, (0, 1, 0, 1), mode='replicate')
            return upsampled

        # Decoder Step 1
        up3 = self.upconv3(conv4)              # Shape: [batch, 4c, H/4, W/4]
        up3 = pad_if_needed(up3, conv3)        # Ensure up3 matches conv3 dimensions
        # Reshape t3 to [batch, 4c, H/4, W/4]
        t_emb3 = t3.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, up3.size(2), up3.size(3))  
        # Alternatively, use expand to save memory:
        # t_emb3 = t3.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, up3.size(2), up3.size(3))
        concat3 = torch.cat([up3, conv3, t_emb3], dim=1)   # Shape: [batch, 12c, H/4, W/4]
        conv5 = self.conv5(concat3, a_onehot, x_cont)         # Shape: [batch, 4c, H/4, W/4]

        # Decoder Step 2
        up2 = self.upconv2(conv5)              # Shape: [batch, 2c, H/2, W/2]
        up2 = pad_if_needed(up2, conv2)        # Ensure up2 matches conv2 dimensions
        # Reshape t2 to [batch, 2c, H/2, W/2]
        t_emb2 = t2.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, up2.size(2), up2.size(3))
        concat2 = torch.cat([up2, conv2, t_emb2], dim=1)   # Shape: [batch, 6c, H/2, W/2]
        conv6 = self.conv6(concat2, a_onehot, x_cont)         # Shape: [batch, 2c, H/2, W/2]

        # Decoder Step 3
        up1 = self.upconv1(conv6)              # Shape: [batch, c, H, W]
        up1 = pad_if_needed(up1, conv1)        # Ensure up1 matches conv1 dimensions
        # Reshape t1 to [batch, c, H, W]
        t_emb1 = t1.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, up1.size(2), up1.size(3))
        concat1 = torch.cat([up1, conv1, t_emb1], dim=1)   # Shape: [batch, 3c, H, W]
        conv7 = self.conv7(concat1, a_onehot, x_cont)         # Shape: [batch, c, H, W]

        # Final Convolution
        return self.final_conv(conv7)          # Shape: [batch, out_channels, H, W]
