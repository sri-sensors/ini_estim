import torch.nn as nn
import torch.nn.functional as F


class MLPDiscriminator(nn.Module):
    def __init__(self, data_dim, encoder_dim, internal_factor=5, start_dim=1):
        """ MLP discriminator 
        
        Parameters
        ----------
        data_dim : int
            The total number of dimensions for the source data. For example,
            if the data represents a sequence that has n_samples and n_feautres,
            then data_dim = n_samples * nfeatures.
        encoder_dim : int
            The total number of dimensions for the encoded data.
        internal_factor : int (Optional)
            The number of nodes in the hidden layer is determined by:
            internal_factor * (data_dim + encoder_dim)
            The default is 5.
        start_dim : int (Optional)
            The start dimension for the discriminator, by default 1. See
            nn.Flatten.
        """
        super().__init__()
        self.data_dim = data_dim
        self.encoder_dim = encoder_dim
        self.internal_dim = (data_dim + encoder_dim)*internal_factor
        self.f_data = nn.Sequential(
            nn.Flatten(start_dim), nn.Linear(data_dim, self.internal_dim)
        )
        self.f_enc = nn.Sequential(
            nn.Flatten(start_dim), nn.Linear(encoder_dim, self.internal_dim)
        )
        self.f_out = nn.Linear(self.internal_dim, 1)
    
    def forward(self, encoded, data):
        tmp = F.relu(self.f_data(data) + self.f_enc(encoded))
        return self.f_out(tmp)
