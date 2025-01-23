import os
import re
import glob
import numpy as np
import torch as th
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    """class representing a PyTorch Multi-Layer-Perceptron"""

    def __init__(self, n_inputs, n_actions):
        """class constructor"""

        # call superclass constructor
        nn.Module.__init__(self)

        # define network architecture
        self.fc1 = nn.Linear(n_inputs, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)
        self.activ_fn = nn.Tanh()
        self.out_activ = nn.Softmax(dim=0)

    def forward(self, x):
        """evaluate the neural network"""

        x = th.from_numpy(x).float()
        x = self.activ_fn(self.fc1(x))
        x = self.activ_fn(self.fc2(x))
        x = self.fc3(x)
        x = np.clip(x.detach().numpy(), -1.0, 1.0)

        return x


def get_model_and_vecnormalize_path(load_path):
    files = os.listdir(load_path)
    if "model.zip" in files:
        model_path = os.path.join(load_path, "model.zip")
        vec_normalize_path = model_path.replace("model.zip", "vecnormalize.pkl")
    else:
        # No best_model.zip, find last model
        files = sorted(glob.glob(os.path.join(load_path, "rl_model*.zip")))

        def extract_number(f):
            s = re.findall("\d+", f)
            return int(s[-1]) if s else -1, f

        model_path = max(files, key=extract_number)
        vec_normalize_path = model_path.replace("rl_model", "vecnormalize").replace(".zip", ".pkl")

    return model_path, vec_normalize_path