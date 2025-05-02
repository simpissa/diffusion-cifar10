import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DiffusionProcess:
    def __init__(self, image_size, channels, hidden_dims=[32, 64, 128], beta_start=1e-4, beta_end=0.02, noise_steps=1000, schedule_type='cosine', device=torch.device('cuda')): # Added schedule_type
        """
        Initialize the diffusion process.
        Args:
            beta_start: Initial noise variance (for linear schedule)
            beta_end: Final noise variance (for linear schedule)
            noise_steps: Number of diffusion steps
            schedule_type: 'linear' or 'cosine'
        """
        super().__init__()
        self.device = device
        self.noise_steps = noise_steps
        self.image_size = image_size
        self.channels = channels

        # --- Schedule Calculation ---
        if schedule_type == 'linear':
            print("Using Linear Noise Schedule")
            self.betas = torch.linspace(beta_start, beta_end, noise_steps, dtype=torch.float32, device=self.device)
        elif schedule_type == 'cosine':
            print("Using Cosine Noise Schedule")
            # Correct Cosine Schedule Calculation (Improved Numerical Stability)
            steps = self.noise_steps + 1
            t = torch.linspace(0, self.noise_steps, steps, dtype=torch.float64) # Use float64 for intermediate calculations
            cosine_s = 8e-3
            alpha_cumprod_t = torch.cos(((t / self.noise_steps) + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2
            alpha_cumprod_t = alpha_cumprod_t / alpha_cumprod_t[0] # Normalize alpha_cumprod_0 = 1

            # Calculate betas from alpha_cumprod
            betas = 1. - (alpha_cumprod_t[1:] / alpha_cumprod_t[:-1])
            # Clip betas to prevent numerical issues (especially near t=0 and t=T)
            self.betas = torch.clip(betas, min=0., max=0.999).to(torch.float32).to(self.device) # Shape [noise_steps]

            # Store the correctly calculated alpha_cumprod (use the full T+1 length)
            self.alpha_cumprod = alpha_cumprod_t.to(torch.float32).to(self.device) # Shape [noise_steps + 1]

        else:
            raise ValueError(f"Unsupported schedule_type: {schedule_type}")

        # --- Calculate Derived Quantities (Common for both schedules) ---
        self.alphas = 1. - self.betas # Shape [noise_steps]

        # Ensure alpha_cumprod is calculated correctly if using linear schedule
        if schedule_type == 'linear':
             alpha_cumprod_from_alphas = torch.cumprod(self.alphas, axis=0)
             # Prepend 1.0 for alpha_cumprod_0
             self.alpha_cumprod = F.pad(alpha_cumprod_from_alphas, (1,0), value=1.0).to(self.device) # Shape [noise_steps + 1]

        # --- Precompute terms needed for sampling and training ---
        # These should work correctly now regardless of schedule, as long as self.betas and self.alpha_cumprod are correct

        # For add_noise q(x_t | x_0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)              # Shape [noise_steps + 1]
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod) # Shape [noise_steps + 1]

        # For posterior q(x_{t-1} | x_t, x_0) - Used in robust sampling step
        # Calculate posterior variance: beta_tilde_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        # Slice alpha_cumprod to get previous timestep values (alpha_bar_{t-1})
        alpha_cumprod_prev = self.alpha_cumprod[:-1] # Shape [noise_steps], indices 0 to T-1
        alpha_cumprod_curr = self.alpha_cumprod[1:]  # Shape [noise_steps], indices 1 to T

        # Use clip to prevent division by zero or negative values if alpha_cumprod is exactly 1
        self.posterior_variance = self.betas * (1.0 - alpha_cumprod_prev) / torch.clamp(1.0 - alpha_cumprod_curr, min=1e-20) # Shape [noise_steps]
        # For numerical stability in sampling variance
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20)) # Shape [noise_steps]

        # Coefficients for posterior mean computation (based on predicting x0):
        # mu_tilde_t = coef1 * x0 + coef2 * xt
        # coef1 = beta_t * sqrt(alpha_bar_{t-1}) / (1 - alpha_bar_t)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(alpha_cumprod_prev) / torch.clamp(1.0 - alpha_cumprod_curr, min=1e-20) # Shape [noise_steps]
        # coef2 = sqrt(alpha_t) * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        self.posterior_mean_coef2 = torch.sqrt(self.alphas) * (1.0 - alpha_cumprod_prev) / torch.clamp(1.0 - alpha_cumprod_curr, min=1e-20) # Shape [noise_steps]


        # --- Initialize Model and Optimizer ---
        self.model = DiffusionModel(image_size, channels, hidden_dims).to(self.device)
        # Ensure optimizer gets the correct learning rate from training function later
        self.optimizer = torch.optim.Adam(self.model.parameters())


    def _extract(self, a, t, x_shape):
        """Extracts coefficients at specified timesteps t and reshapes for broadcasting."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))) # Reshape to [batch_size, 1, 1, 1]

    def add_noise(self, x_start, t):
        """
        Forward process q(x_t | x_start=x_0): Add noise to the input images.
        Args:
            x_start: Clean images tensor of shape [batch_size, channels, height, width]
            t: Timesteps tensor of shape [batch_size] (indices 0 to noise_steps-1)
        Returns:
            Tuple of (noisy_images x_t, noise epsilon)
        """
        # Noise sampled from standard Gaussian
        noise = torch.randn_like(x_start, device=self.device)

        # Get sqrt(alpha_bar_t) and sqrt(1 - alpha_bar_t) using t
        # Note: t corresponds to step t, so we need index t in alpha_cumprod
        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alpha_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alpha_cumprod, t, x_start.shape)

        # Calculate noisy image x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        noisy_images = sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise
        return noisy_images, noise

    @torch.no_grad()
    def sample(self, num_samples=16):
        """
        Reverse process p(x_{t-1} | x_t): Generate new samples by reversing the diffusion.
        Args:
            num_samples: Number of samples to generate
        Returns:
            Generated images tensor (likely in [-1, 1] range)
        """
        self.model.eval() # Set model to evaluation mode

        # Start with random noise x_T at the final timestep
        shape = (num_samples, self.channels, self.image_size[0], self.image_size[1])
        x_t = torch.randn(shape, device=self.device)

        # Iterate backwards from T-1 down to 0
        for t_int in reversed(range(self.noise_steps)):
            # Create a tensor for the current timestep t
            t = torch.full((num_samples,), t_int, device=self.device, dtype=torch.long)

            # Predict the noise added at this timestep using the model
            predicted_noise = self.model(x_t, t)

            # Get necessary coefficients using helper function _extract
            # Note: Coefficients often correspond to step t, matching index t in betas, alphas, posterior coeffs
            #       Alpha cumulative products use index t directly.
            sqrt_alpha_cumprod_t = self._extract(self.sqrt_alpha_cumprod, t, x_t.shape)
            sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alpha_cumprod, t, x_t.shape)
            posterior_mean_coef1_t = self._extract(self.posterior_mean_coef1, t, x_t.shape)
            posterior_mean_coef2_t = self._extract(self.posterior_mean_coef2, t, x_t.shape)
            posterior_log_variance_t = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)

            # Estimate x0 based on predicted noise (Equation 11 in DDPM)
            # x0_pred = (x_t - sqrt(1 - alpha_bar_t) * noise_pred) / sqrt(alpha_bar_t)
            x0_pred = (x_t - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / torch.clamp(sqrt_alpha_cumprod_t, min=1e-20)
            # Optional: Clamp the predicted x0 to the valid data range [-1, 1]
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            # Calculate the mean of the posterior q(x_{t-1} | x_t, x_0_pred) (Equation 7 in DDPM)
            # posterior_mean = coef1 * x0_pred + coef2 * x_t
            posterior_mean = posterior_mean_coef1_t * x0_pred + posterior_mean_coef2_t * x_t

            # Get noise for sampling x_{t-1}
            noise = torch.randn_like(x_t)
            if t_int == 0:
                 # No noise added at the final step (t=0)
                 x_t = posterior_mean
            else:
                 # Add noise scaled by the posterior variance
                 # x_{t-1} = posterior_mean + sqrt(posterior_variance) * noise
                 # variance = exp(log_variance) -> sqrt(variance) = exp(0.5 * log_variance)
                 x_t = posterior_mean + (0.5 * posterior_log_variance_t).exp() * noise

        self.model.train() # Set model back to training mode
        # Final output x_0 should be in the data range (e.g., [-1, 1])
        return x_t

    def train_step(self, x_start):
        """
        Perform one training step for the diffusion model.
        Args:
            x_start: Clean images tensor of shape [batch_size, channels, height, width] in range [-1, 1]
        Returns:
            Loss value (scalar) for the step
        """
        self.optimizer.zero_grad()

        # 1. Sample random timesteps t for each image in the batch
        # Indices from 0 to noise_steps-1
        batch_size = x_start.shape[0]
        t = torch.randint(0, self.noise_steps, (batch_size,), device=self.device, dtype=torch.long)

        # 2. Add noise to images to get x_t using the forward process q(x_t | x_0)
        noisy_images, added_noise = self.add_noise(x_start, t) # added_noise is epsilon

        # 3. Predict the noise (epsilon) using the U-Net model
        predicted_noise = self.model(noisy_images, t)

        # 4. Calculate loss between predicted noise and actual added noise
        # Use MSE loss as is common (L2 loss)
        # Other options: L1 loss (SmoothL1Loss)
        loss = F.mse_loss(predicted_noise, added_noise)
        # loss = F.smooth_l1_loss(predicted_noise, added_noise) # Alternative L1

        # 5. Perform backpropagation and optimizer step
        loss.backward()
        # Optional: Gradient clipping if gradients explode
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # 6. Return the loss value
        return loss.item()


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Module for generating sinusoidal position embeddings for the timestep.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Forward pass for generating time embeddings.
        Args:
            time: Tensor of shape [batch_size] representing timesteps.
        Returns:
            Tensor of shape [batch_size, dim] representing embeddings.
        """
        device = torch.device('cuda')
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # Handle odd dimensions
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0,1)) # Pad the last dimension
        return embeddings
    
class Block(nn.Module):
    """
    Basic convolutional block for U-Net. Consists of Conv2d, GroupNorm, and SiLU activation.
    """
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU() # Swish activation

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x
    
class ResnetBlock(nn.Module):
    """
    Residual block incorporating time embeddings.
    """
    def __init__(self, in_channels, out_channels, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
            if time_emb_dim is not None
            else None
        )

        self.block1 = Block(in_channels, out_channels, groups=groups)
        self.block2 = Block(out_channels, out_channels, groups=groups)
        # Use 1x1 conv if input and output channels differ for the residual connection
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if self.mlp is not None and time_emb is not None:
            # Project time embedding and add to the feature map
            time_emb = self.mlp(time_emb)
            # Need to reshape time_emb: [batch, channels] -> [batch, channels, 1, 1]
            h = h + time_emb[:, :, None, None]

        h = self.block2(h)
        return h + self.res_conv(x) # Add residual connection
    

class DiffusionModel(nn.Module):
    def __init__(self, image_size, channels, hidden_dims=[32, 64, 128], num_res_blocks=2, groups=8):
        """
        Initialize the diffusion model.
        Args:
            config: Configuration object with model parameters
        """
        super().__init__()
        
        # TODO: Check the parameters and save up necessary ones
        
        # TODO: Implement the time embedding module
        # Create a time embedding MLP to encode the timestep
        # This should consist of linear layers with SiLU activation
        time_dim = hidden_dims[0] * 4 # A common choice for the internal dim of time MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dims[0]), # Map time to initial embedding dim
            nn.Linear(hidden_dims[0], time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim) # Final projection dim, used in ResNet blocks
        )
        effective_time_emb_dim = time_dim

        # TODO: Implement the initial convolution layer
        # Create an initial convolution layer to process the input image
        self.init_conv = nn.Conv2d(channels, hidden_dims[0], kernel_size=3, padding=1)

        # TODO: Implement the encoder (downsampling path)
        # Create a list of down blocks for the encoder path
        # Each block should include convolutions, batch normalization, and activation
        # Don't forget to include a downsampling mechanism (e.g., MaxPool2d)
        self.down_blocks = nn.ModuleList([])
        channels_list = [hidden_dims[0]]
        now_channels = hidden_dims[0]
        for level, mult in enumerate(hidden_dims):
            out_ch = mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResnetBlock(
                        now_channels,
                        out_ch,
                        time_emb_dim=effective_time_emb_dim,
                        groups=groups
                    )
                )
                now_channels = out_ch
                channels_list.append(now_channels)
            # Downsample at the end of each level, except the last one
            if level != len(hidden_dims) - 1:
                self.down_blocks.append(nn.Conv2d(now_channels, now_channels, kernel_size=4, stride=2, padding=1)) # Downsample
                channels_list.append(now_channels)

        # TODO: Implement the bottleneck
        # Create a bottleneck block with additional processing
        mid_channels = channels_list[-1]
        self.mid_block1 = ResnetBlock(mid_channels, mid_channels, time_emb_dim=effective_time_emb_dim, groups=groups)
        self.mid_block2 = ResnetBlock(mid_channels, mid_channels, time_emb_dim=effective_time_emb_dim, groups=groups)

        # TODO: Implement the decoder (upsampling path)
        # Create a list of up blocks for the decoder path
        # Each block should include upsampling, concatenation with skip connections,
        # and convolutions with batch normalization and activation
        self.up_blocks = nn.ModuleList([])
        for level, mult in reversed(list(enumerate(hidden_dims))):
            out_ch = mult
            for _ in range(num_res_blocks + 1): # +1 to account for the skip connection input
                # Input channels: current channels + skip connection channels
                in_ch = channels_list.pop() # Get channel count from corresponding down block
                self.up_blocks.append(
                    ResnetBlock(
                        now_channels + in_ch, # Concatenated channels
                        out_ch,
                        time_emb_dim=effective_time_emb_dim,
                        groups=groups
                    )
                )
                now_channels = out_ch
            # Upsample at the end of each level, except the first one (innermost)
            if level != 0:
                # Use ConvTranspose2d for upsampling
                self.up_blocks.append(nn.ConvTranspose2d(now_channels, now_channels, kernel_size=4, stride=2, padding=1))

        # TODO: Implement time embedding projections
        # Create projections for injecting time features into each decoder layer



        # TODO: Implement the final output layer
        # Create a final convolution to map to the output channels
        self.final_conv = nn.Sequential(
            nn.GroupNorm(groups, hidden_dims[0]), # Normalize before final activation
            nn.SiLU(),
            nn.Conv2d(hidden_dims[0], channels, kernel_size=3, padding=1)
        )


    def forward(self, x, t):
        """
        Forward pass through the U-Net model.
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            t: Timesteps tensor of shape [batch_size]
        Returns:
            Tensor of shape [batch_size, channels, height, width]
        """
        # TODO: Implement the forward pass
        # 1. Embed the timestep
        # 2. Process input through initial convolution
        # 3. Store residuals for skip connections
        # 4. Process through encoder blocks
        # 5. Process through bottleneck
        # 6. Process through decoder blocks with time injection and skip connections
        # 7. Apply final convolution
        # 8. Return the output tensor
        # 1. Embed the timestep
        t = self.time_mlp(t) if self.time_mlp is not None else None

        # 2. Process input through initial convolution
        x = self.init_conv(x)
        initial_x = x.clone() # Save for final skip connection if needed (optional, common in some variants)

        # 3. Store residuals (skip connections) during the downsampling path
        residuals = [x]
        h = x

        # 4. Process through encoder blocks
        for block in self.down_blocks:
            if isinstance(block, ResnetBlock):
                h = block(h, t)
            else: # Downsampling layer
                h = block(h)
            residuals.append(h) # Store output of each block/downsample layer

        # 5. Process through bottleneck
        h = self.mid_block1(h, t)
        h = self.mid_block2(h, t)

        # 6. Process through decoder blocks with time injection and skip connections
        for block in self.up_blocks:
            if isinstance(block, ResnetBlock):
                # Get the corresponding residual
                res = residuals.pop()
                # Concatenate along the channel dimension
                h = torch.cat([h, res], dim=1)
                h = block(h, t)
            else: # Upsampling layer
                h = block(h)

        # Concatenate with the output of the *initial* conv layer (often done)
        # This depends slightly on the specific U-Net variant. Here, we assume the final ResNet block in the decoder
        # already brought the channel count down to model_channels to match init_conv output.
        # If not, adjust the final_conv input channels or the last up_block's output channels.
        # h = torch.cat([h, initial_x], dim=1) # Optional: Add initial skip connection

        # 7. Apply final convolution
        # Adjust final_conv input if initial_x skip was used: nn.Conv2d(model_channels + model_channels, ...)
        return self.final_conv(h)

