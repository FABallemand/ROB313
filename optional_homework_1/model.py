import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block
    """

    def __init__(self):
        super(ResidualBlock, self).__init__()

        self._conv1 = nn.Conv2d(256, 256, 3, 1, "same")
        self._conv2 = nn.Conv2d(256, 256, 1, 1, "same")

    def forward(self, x):
        x_out = self._conv1(x)
        x_out = F.relu(x_out)
        x_out = self._conv2(x_out)
        x_out = F.relu(x_out)
        return x + x_out
    

class ResidualTransposeBlock(nn.Module):
    """
    Residual transpose block
    """

    def __init__(self):
        super(ResidualTransposeBlock, self).__init__()

        self._t_conv1 = nn.ConvTranspose2d(256, 256, 3, 1, 1)
        self._t_conv2 = nn.ConvTranspose2d(256, 256, 1, 1, 0)

    def forward(self, x):
        x_out = self._t_conv1(x)
        x_out = F.relu(x_out)
        x_out = self._t_conv2(x_out)
        x_out = F.relu(x_out)
        return x + x_out
    

class VectorQuantizer(nn.Module):
    """
    Vector quantizer

    Inspired by: https://nbviewer.org/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # Convert inputs from [B,C,H,W] to [B,H,W,C]
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t())) # [BHW, K]

        # Encode input
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) # [BHW, 1]
        encodings = torch.zeros(encoding_indices.shape[0],
                                self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1) # [BHW, K]

        # Quantize and unflatten encoded input
        quantized = torch.matmul(encodings, self._embedding.weight) # [BHW, D]
        quantized = quantized.view(input_shape) # [B, H, W, D]

        # Compute loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs) # Commitment loss
        q_latent_loss = F.mse_loss(quantized, inputs.detach()) # Embedding loss
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        # avg_probs = torch.mean(encodings, dim=0)
        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Convert quantized from [B,H,W,C] to [B,C,H,W]
        return quantized.permute(0, 3, 1, 2).contiguous(), loss


class VQVAE(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._input_size = torch.Size([1, 28, 28])

        self._num_embeddings=512
        self._embedding_dim=64
        self._commitment_cost = 0.25

        self.reconstruction_criterion = nn.MSELoss()

        # Encoder
        self._encoder = nn.Sequential(
            nn.Conv2d(self._input_size[0], 256, 4, 2),
            nn.Conv2d(256, 256, 4, 2),
            ResidualBlock(),
            ResidualBlock()
        )

        # Pre-vector quantisation convolution
        self._pre_vq_conv = nn.Conv2d(256, 256, 1, 1)
        
        # Vector quantisation module
        self._vq = VectorQuantizer(self._num_embeddings, self._embedding_dim,
                                   self._commitment_cost)

        # Decoder
        self._decoder = nn.Sequential(
            ResidualTransposeBlock(),
            ResidualTransposeBlock(),
            nn.ConvTranspose2d(256, 256, 4, 2, 0, 1),
            nn.ConvTranspose2d(256, self._input_size[0], 4, 2, 0, 0)
        )

    def loss_function(self, x_recon, x, vq_loss):
        reconstruction_loss = self.reconstruction_criterion(x_recon, x)
        loss = reconstruction_loss + vq_loss
        return {"loss": loss,
                "reconstruction_loss": reconstruction_loss,
                "vq_loss": vq_loss}

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        quantized, vq_loss = self._vq(z)
        x_recon = self._decoder(quantized)

        return x_recon, z, quantized, vq_loss
    
    def encode(self, x):
        result = self.encoder(x)
        return result

    def decode(self, z):
        result = self.decoder(z)
        return result
