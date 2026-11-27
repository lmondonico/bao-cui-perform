from torch import nn

class BaseMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation=nn.SiLU(), residual=False, last_act=False):
        super(BaseMLP, self).__init__()
        self.residual = residual
        if residual:
            assert output_dim == input_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, output_dim),
            activation if last_act else nn.Identity()
        )
    def forward(self, x):
        return x + self.mlp(x) if self.residual else self.mlp(x)

class BaseMLP_2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layer=4, activation=nn.SiLU(), residual=False, last_act=False):
        super(BaseMLP_2, self).__init__()
        self.residual = residual
        self.num_layer = num_layer
        if residual:
            assert output_dim == input_dim
        self.mlps = nn.ModuleList()
        mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            activation
        )
        self.mlps.append(mlp)
        for _ in range(self.num_layer-2):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                activation
            )
            self.mlps.append(mlp)
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim) if not last_act else nn.Identity(),
            activation if last_act else nn.Identity()
        )
        self.mlps.append(mlp)
        
    def forward(self, x):
        if self.residual:
            for mlp in self.mlps:
                x = x + mlp(x)
        else:
            for mlp in self.mlps:
                x = mlp(x)
        return x