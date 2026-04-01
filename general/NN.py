import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryClassifier(nn.Module):
    """Fully-connected binary classifier with optional batch-norm & dropout.

    A stack of num_layers hidden Linear → Activation → BatchNorm
    blocks, followed by a 1-unit output layer and a sigmoid.

    Args:
        input_dim : int
            Number of input features.
        hidden_dim : int
            Width of every hidden layer.
        num_layers : int
            Number of hidden layers.
        activation : callable, default F.relu
            Non-linearity applied after each hidden Linear.
        dropout : float, default 0.0
            Dropout probability applied after batch-norm.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation=F.relu,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        # choose weight initialiser based on non‑linearity
        if activation == F.relu:
            init_fn = nn.init.kaiming_normal_
            nonlin = "relu"
        elif activation == F.tanh:
            init_fn = nn.init.xavier_normal_
            nonlin = "tanh"
        else:  # fallback
            init_fn = nn.init.kaiming_uniform_
            nonlin = "linear"

        # build hidden stack
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layer = nn.Linear(in_dim, hidden_dim)
            init_fn(layer.weight)
            nn.init.zeros_(layer.bias)
            self.hidden_layers.append(layer)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # output layer
        self.output_layer = nn.Linear(hidden_dim, 1)
        init_fn(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return P(y=1 | x) for a mini-batch.

        Args:
            x : Tensor, shape (batch, input_dim)

        Returns: 
            Tensor shape (batch,) – probabilities in [0, 1].
        """
        for layer, bn in zip(self.hidden_layers, self.batch_norms):
            x = self.activation(layer(x))
            x = bn(x)
            x = self.dropout(x)
        logits = self.output_layer(x).squeeze(-1)
        return torch.sigmoid(logits)



class PoissonRegressor(nn.Module):
    """
    Multi-layer Poisson regression network **with offset support**.

    • *input_dim*  – number of true covariates **(does not include offset)**  
    • *num_layers* – total hidden layers (≥ 1).  
    • *hidden_dim* – width of each hidden layer.  
    • *activation* – name of an activation class in torch.nn (e.g. "ReLU").  
    • *dropout*    – dropout probability applied after every activation.

    Forward signature
    -----------------
    >>> lam = net(x_batch, offset_batch)
    • x_batch      : Tensor, shape (batch, input_dim)  
    • offset_batch : 1-D Tensor, same batch length  
      (typically exposure or baseline time)  

    Returns
    -------
    lam : Tensor, shape (batch,) – strictly positive rate parameters λ.
    """

    def __init__(
        self,
        *,
        input_dim:  int,
        num_layers: int,
        hidden_dim: int,
        activation: str = "ReLU",
        dropout:    float = 0.0,
    ):
        super().__init__()

        # -- activation ----------------------------------------------------
        if isinstance(activation, str):
            act_cls = getattr(nn, activation)
        elif isinstance(activation, type) and issubclass(activation, nn.Module):
            act_cls = activation
        else:
            raise ValueError("activation must be nn.Module subclass or its name")

        # -- hidden stack --------------------------------------------------
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.hidden = nn.Sequential(*layers)

        # -- final output layer  (produces log-rate) -----------------------
        self.out = nn.Linear(hidden_dim, 1)

        # optional: initialise weights sensibly
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    # --------------------------------------------------------------------
    def forward(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x      : (batch, input_dim)  – covariates (without offset)
        offset : (batch,)           – exposure vector (must be > 0)

        Returns
        -------
        lam : (batch,) – Poisson rate λ = offset · exp(f(x))
        """
        h = self.hidden(x)
        log_rate = self.out(h).squeeze(-1)     # (batch,)
        return offset * torch.exp(log_rate)    # strictly positive
