import torch
import torch.nn as nn


class AESurrogateModel():
    def __init__(self, population: int):
        self.model = torch.load('models/autoencoder_barabasi_100k.pt',
                                weights_only=False)
        self.model.eval()

    def simulate(self, alpha, beta, gamma=None, delta=None, init_inf_frac=None, tmax=None):
        alpha_t = torch.tensor(float(alpha), dtype=torch.float32)
        beta_t = torch.tensor(float(beta), dtype=torch.float32)
        self.daily_incidence = self.model(
            torch.tensor([beta_t, alpha_t])).detach().cpu().numpy()
        return self.daily_incidence


class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, output_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_size, latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_size, output_size),
        )

    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        x_recon = self.decoder(z)
        return x_recon

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, output_size):
        super(VariationalAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
        )

        self.enc_mu = nn.Linear(hidden_size, latent_size)
        self.enc_logvar = nn.Linear(hidden_size, latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_size, output_size)
        )

        self.max_logvar = nn.Parameter(torch.ones(
            latent_size) * 0.5, requires_grad=True)
        self.min_logvar = nn.Parameter(torch.ones(
            latent_size) * -0.5, requires_grad=True)

    def encode(self, x):
        x = self.encoder(x)
        mu = self.enc_mu(x)
        log_var = self.enc_logvar(x)
        return mu, log_var

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=self.min_logvar, max=self.max_logvar)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z

    def decode(self, z):
        x = self.decoder(z)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
