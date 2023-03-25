import torch.nn as nn
import torch.nn.functional as F
import torch
from .base import BaseModel, Setting


class SensorEncoder(BaseModel):
    ALGORITHMS = [
        'binary', 'linear', 'scaled_linear', 'biomimetic1', 'biomimetic2'
        ]

    def __init__(self, amin, amax, fmin, fmax, algo='binary', *args, **kwargs):
        """ Sensor Encoder according to [1]
        
        Parameters
        ----------
        amin : float
        amax : float
            Min/max stimulation amplitudes
        fmin : float
        fmax : float
            Min/max stimulation frequencies
        fsample : float
            Sample rate 
        algo : str
            Can be any of SensorEncoder.ALGORITHMS:
            binary - constant f=fmin and a=amin if input is above a threshold,
                otherwise 0.
            linear - f and a are linearly scaled along fmin/fmax and amin/amax
            scaled_linear - same as linear but input is pre-multipled by 2
            biomimetic1 - f and a take velocity and acceleration into account
            biomimetic2 - f takes vel/acc into account, a is fixed to amin. Note
                also that biomimetic2 has fixed weights (from [1]), and so will
                not follow fmin/fmax, although f will be clipped to positive
                values.

        Notes
        -----
        For all algorithms, the input is assumed to be normalized to 0 to 1.
        Except for biomimetic2, no clipping is performed.

        References
        ----------
        [1] George, Jacob A., et al. "Biomimetic sensory feedback through 
            peripheral nerve stimulation improves dexterous use of a bionic 
            hand." Science Robotics 4.32 (2019): eaax2352.
        """
        super().__init__(*args, **kwargs)
        self.amin = Setting(amin)
        self.amax = Setting(amax)
        self.fmin = Setting(fmin)
        self.fmax = Setting(fmax)
        self.algo = Setting(algo)
        self.threshold = Setting(0.01)
        self.kdiff = nn.Parameter(
            torch.FloatTensor([-0.5, 0.0, 0.5]).view(1, 1, -1),
            requires_grad=False
        )
        self.configure()

    @classmethod
    def from_checkpoint(cls, checkpoint):
        cfg = checkpoint["config"]
        out = cls(cfg["amin"], cfg["amax"], cfg["fmin"], cfg["fmax"], cfg["algo"])
        out.threshold = cfg["threshold"]
        return out
    
    def configure(self):
        self.net = self._get_net(self.algo)
    
    def _get_net(self, algo):
        if algo == 'binary':
            return self._binary
        elif algo == 'linear':
            return self._linear
        elif algo == 'scaled_linear':
            return self._scaled_linear
        elif algo == 'biomimetic1':
            return self._biomimetic1
        elif algo == 'biomimetic2':
            return self._biomimetic2
        else:
            raise ValueError("Invalid algorithm \"{}\"".format(algo))
        
    def forward(self, x):
        """ Compute electrode stimulation from the input 
        
        Parameters
        ----------
        x : FloatTensor
            Input sensor values. These should be normalized to 0-1, and dims
            should follow (batch_size, num_samples, num_sensors)

        Returns
        -------
        frequency (FloatTensor), amplitude (FloatTensor)
        """
        return self.net(x)

    def _binary(self, x):
        f = torch.zeros_like(x)
        a = torch.zeros_like(x)
        idx = x > self.threshold
        f[idx] = self.fmin
        a[idx] = self.amin
        return f, a
    
    def _linear(self, x):
        f = x*(self.fmax - self.fmin) + self.fmin
        a = x*(self.amax - self.amin) + self.amin
        return f, a
    
    def _scaled_linear(self, x):
        return self._linear(2.0*x)
    
    def _biomimetic1(self, x):
        nchannels = x.shape[2]
        kdiff = self.kdiff.expand(nchannels, -1, -1)
        x = x.transpose(1, 2)
        vel = F.conv1d(F.pad(x, (1, 1), mode='replicate'), kdiff, groups=nchannels)
        acc = F.conv1d(F.pad(vel, (1, 1), mode='replicate'), kdiff, groups=nchannels)
        alpha = 5.0*torch.relu(vel) + x
        f = alpha*(self.fmax - self.fmin) + self.fmin
        a = alpha*(self.amax - self.amin) + self.amin
        f = f.transpose(1, 2).contiguous()
        a = a.transpose(1, 2).contiguous()
        return f, a

    def _biomimetic2(self, x):
        nchannels = x.shape[2]
        kdiff = self.kdiff.expand(nchannels, -1, -1)
        x = x.transpose(1, 2)
        vel = F.conv1d(F.pad(x, (1, 1), mode='replicate'), kdiff, groups=nchannels)
        acc = F.conv1d(F.pad(vel, (1, 1), mode='replicate'), kdiff, groups=nchannels)
        kp = torch.tensor([-185.0, 186.0], dtype=torch.float32).view(1,1,-1).expand(nchannels, -1, -1)
        kv = torch.tensor([-109.0, -360.0, 1559.0], dtype=torch.float32).view(1,1,-1).expand(nchannels, -1, -1)
        ka = torch.tensor([170.0, 364.0], dtype=torch.float32).view(1,1,-1).expand(nchannels, -1, -1)
        x = F.conv1d(F.pad(x, (1, 0), mode='replicate'), kp, groups=nchannels)
        vel = F.conv1d(F.pad(vel, (2, 0), mode='replicate'), kv, groups=nchannels)
        acc = F.conv1d(F.pad(acc, (1, 0), mode='replicate'), ka, groups=nchannels)
        
        f = (x + vel + acc - 3.0)
        f = f.transpose(1, 2).contiguous()
        a = torch.full_like(f, self.amin)
        return f, a


class BiomimeticEncoder(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure()

    def configure(self):
        self.net = SensorEncoder(0.0, 1.0, 0.0, 1.0, 'biomimetic1')
    
    def forward(self, x):
        return self.net(x)[1]
