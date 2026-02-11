import os
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import tinycudann as tcnn

##################################################  SIREN MLP ##################################################
# SIREN (MLP with sinusoid activation function) is based on the paper:
# "Implicit Neural Representations with Periodic Activation Functions" # https://arxiv.org/abs/2006.09661
# Code is available from this repository: https://www.vincentsitzmann.com/siren/

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output


##################################################  ReLU MLP with Positional Encoding ##################################################
# NeRF positional encoding is based on the paper:
# "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" https://arxiv.org/abs/2003.08934

# Fourier positional encoding is based on the paper:
# "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains" https://arxiv.org/abs/2006.10739
# The code for Fourier positional encoding can be found here: https://bmild.github.io/fourfeat/

# Hash positional encoding is based on the paper:
# "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding" https://arxiv.org/abs/2201.05989

class PositionalEncoding(nn.Module):
    """
    Flexible positional encoding:
    - NeRF-style (sin/cos with 2^k frequencies)
    - Gaussian Fourier features (random projection matrix B)
    """
    def __init__(self, in_features, num_frequencies=10, include_input=True,
                 encoding_type="nerf", gauss_scale=10.0):
        super().__init__()
        self.in_features = in_features
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.encoding_type = encoding_type

        if encoding_type == "nerf":
            # NeRF-style frequency bands: [1, 2, 4, ..., 2^(L-1)]
            self.freq_bands = 2. ** torch.linspace(0., num_frequencies - 1, num_frequencies)
            self.register_buffer("freq_bands_buf", self.freq_bands)

        elif encoding_type == "gaussian":
            # Gaussian Fourier features: random projection matrix B
            B = torch.randn(num_frequencies, in_features) * gauss_scale
            self.register_buffer("B", B)

        else:
            raise ValueError("encoding_type must be 'nerf' or 'gaussian'")

    def forward(self, x):
        """
        Args:
            x: Tensor [B, in_features]
        Returns:
            encoded: Tensor [B, out_dim]
        """
        out = [x] if self.include_input else []

        if self.encoding_type == "nerf":
            for freq in self.freq_bands_buf:
                out.append(torch.sin(freq * x))
                out.append(torch.cos(freq * x))

        elif self.encoding_type == "gaussian":
            x_proj = 2. * torch.pi * (x @ self.B.T)   # [B, num_frequencies]
            out.append(torch.sin(x_proj))
            out.append(torch.cos(x_proj))

        return torch.cat(out, dim=-1)


class PosEncMLP(nn.Module):
    """
    ReLU-based MLP with positional encoding (NeRF or Gaussian) and Xavier init
    """
    def __init__(self, in_features, out_features, hidden_features, hidden_layers,
                 num_encoding_freqs=10, include_input=True, outermost_linear=True,
                 encoding_type="nerf", gauss_scale=10.0):
        super().__init__()

        # Positional encoding
        self.pos_enc = PositionalEncoding(
            in_features,
            num_frequencies=num_encoding_freqs,
            include_input=include_input,
            encoding_type=encoding_type,
            gauss_scale=gauss_scale
        )

        # Figure out dimension after encoding
        if encoding_type == "nerf":
            pe_out_dim = in_features * (2 * num_encoding_freqs + (1 if include_input else 0))
        elif encoding_type == "gaussian":
            pe_out_dim = (2 * num_encoding_freqs + (in_features if include_input else 0))

        # Build MLP layers
        layers = [nn.Linear(pe_out_dim, hidden_features), nn.ReLU(inplace=True)]
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.ReLU(inplace=True))

        if outermost_linear:
            layers.append(nn.Linear(hidden_features, out_features))
        else:
            layers.append(nn.Linear(hidden_features, out_features))
            layers.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, coords):
        encoded = self.pos_enc(coords)
        return self.net(encoded)


class HashMLP(nn.Module):
    def __init__(self, in_features=2, out_features=3, hidden_features=256, hidden_layers=3):
        super().__init__()
        encoding_config = {
            "otype": "Grid",
            "type": "Hash",
            "n_levels": 12,
            "n_features_per_level": 2,
            "log2_hashmap_size": 16,
            "base_resolution": 4,
            "per_level_scale": 1.61,
            "interpolation": "Linear"
        }
        self.encoder = tcnn.Encoding(n_input_dims=in_features, encoding_config=encoding_config)

        layers = [nn.Linear(12 * 2, hidden_features), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_features, hidden_features), nn.ReLU()]
        layers += [nn.Linear(hidden_features, out_features)]

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x_encoded = self.encoder(x).float()
        return self.mlp(x_encoded)


##################################################  Homography as Motion Model ##################################################
# For writing this function, we referred to the following repository: https://github.com/shnnam/nir

class Homography(nn.Module):
    def __init__(self, in_features=1, hidden_features=256, hidden_layers=1):
        super().__init__()
        out_features = 8

        self.net = []
        self.net.append(nn.Linear(in_features, hidden_features))
        self.net.append(nn.ReLU(inplace=True))
        for i in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.ReLU(inplace=True))
        self.net.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*self.net)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            # Initialize the bias of the last layer to represent an identity homography
            self.net[-1].bias.copy_(torch.Tensor([1., 0., 0., 0., 1., 0., 0., 0.]))

    def forward(self, coords):
        output = self.net(coords)
        return output

def apply_homography(x, h):
    # Extend homography vector to a full 3x3 matrix by adding ones
    h = torch.cat([h, torch.ones_like(h[:, [0]])], -1)

    # Reshape to batch of 3x3 matrices
    h = h.view(-1, 3, 3)

    # Convert input coordinates to homogeneous coordinates by adding 1
    x = torch.cat([x, torch.ones_like(x[:, 0]).unsqueeze(-1)], -1).unsqueeze(-1)

    # Apply homography transformation using batch matrix multiplication
    o = torch.bmm(h, x).squeeze(-1)

    # Convert back from homogeneous coordinates to 2D coordinates
    o = o[:, :-1] / o[:, [-1]]

    return o



##################################################  Basic Functions ##################################################
# For writing these functions, we referred to the following repository: https://github.com/shnnam/nir

def get_mgrid(sidelen, vmin=-1, vmax=1):
    # If vmin and vmax are not lists, convert them to lists with repeated values
    if type(vmin) is not list:
        vmin = [vmin for _ in range(len(sidelen))]
    if type(vmax) is not list:
        vmax = [vmax for _ in range(len(sidelen))]

    # Generate linearly spaced points between vmin and vmax for each dimension
    tensors = tuple([torch.linspace(vmin[i], vmax[i], steps=sidelen[i]) for i in range(len(sidelen))])

    # Create a meshgrid from the tensors and stack them into a single tensor
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)

    # Reshape the grid to have shape (number of points, number of dimensions)
    mgrid = mgrid.reshape(-1, len(sidelen))

    return mgrid

def jacobian(y, x):
    B, N = y.shape
    jacobian = list()
    for i in range(N):
        v = torch.zeros_like(y)
        v[:, i] = 1.
        dy_i_dx = torch.autograd.grad(y,
                                      x,
                                      grad_outputs=v,
                                      retain_graph=True,
                                      create_graph=True)[0]  # shape [B, N]
        jacobian.append(dy_i_dx)
    jacobian = torch.stack(jacobian, dim=1).requires_grad_()
    return jacobian


class VideoFitting(Dataset):
    def __init__(self, path, transform=None):
        super().__init__()

        self.path = path

        # Use a default transform (ToTensor) if none is provided
        if transform is None:
            self.transform = ToTensor()
        else:
            self.transform = transform

        # Load the video into a tensor
        self.video = self.get_video_tensor()

        # Extract the number of frames, height, and width from the video tensor
        self.num_frames, _, self.H, self.W = self.video.size()

        # Reshape the video into pixels and generate grid coordinates for each pixel
        self.pixels = self.video.permute(2, 3, 0, 1).contiguous().view(-1, 3)
        self.coords = get_mgrid([self.H, self.W, self.num_frames])

        # Shuffle the pixels and coordinates for randomization
        shuffle = torch.randperm(len(self.pixels))
        self.pixels = self.pixels[shuffle]
        self.coords = self.coords[shuffle]

    def get_video_tensor(self):

        # Load and sort the frame files from the specified path

        # For 2D Scene Dataset
        frames = [f for f in os.listdir(self.path) if f.endswith('.png')]
        frames = sorted(frames, key=lambda x: int(x.split('_')[0]))
        print(frames)
        video = []

        # Load each frame as an image, apply the transform, and append to the video list
        for i in range(len(frames)):
        # for i in range(0, 1):
            img = Image.open(os.path.join(self.path, frames[i]))
            img = self.transform(img)
            video.append(img)

        # Stack the frames into a 4D tensor with shape (num_frames, channels, height, width)
        return torch.stack(video, 0)

    def __len__(self):
        # Return the length of the dataset, hardcoded to 1 (single batch)
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError  # Only allow idx=0, raise an error otherwise

        # Return the shuffled coordinates and corresponding pixel values
        return self.coords, self.pixels
    





