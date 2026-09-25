"""PyTorch demand forecasting model.

A small feed-forward network over the same engineered feature set used by
XGBoost (calendar, lag, rolling, price/discount features), with learned
embeddings for the two categorical product attributes. Sharing the feature
set with the tree model isolates the comparison to "does model family
matter" rather than "who got better features."
"""
import torch
from torch import nn


class DemandMLP(nn.Module):
    def __init__(self, n_categories: int, n_subcategories: int, n_numeric: int,
                 cat_emb_dim: int = 4, subcat_emb_dim: int = 8, hidden_dims=(128, 64)):
        super().__init__()
        self.category_emb = nn.Embedding(n_categories, cat_emb_dim)
        self.subcategory_emb = nn.Embedding(n_subcategories, subcat_emb_dim)

        input_dim = cat_emb_dim + subcat_emb_dim + n_numeric
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)]
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, category_idx: torch.Tensor, subcategory_idx: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.category_emb(category_idx), self.subcategory_emb(subcategory_idx), numeric], dim=1)
        return self.mlp(x).squeeze(-1)
