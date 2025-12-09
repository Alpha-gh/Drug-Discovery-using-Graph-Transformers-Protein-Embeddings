# cmgt_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierDistanceEncoding(nn.Module):
    def __init__(self, d_model, num_bands=16, max_freq=10.0):
        super().__init__()
        self.num_bands = num_bands
        self.max_freq = max_freq
        self.proj = nn.Linear(2 * num_bands, d_model)

    def forward(self, coords, mask):
        B, N, _ = coords.shape
        c1 = coords.unsqueeze(2)
        c2 = coords.unsqueeze(1)
        d = torch.norm(c1 - c2, dim=-1)

        m = mask.unsqueeze(1) * mask.unsqueeze(2)
        d = d * m.float()

        denom = m.sum(-1).clamp(min=1.0)
        agg = d.sum(-1) / denom

        freqs = torch.linspace(
            1.0, self.max_freq, self.num_bands, device=coords.device
        ).view(1, 1, -1)
        v = agg.unsqueeze(-1) * freqs
        feats = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)
        return self.proj(feats)


class SimpleTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead=4, dim_ff=512):   # nhead 8 -> 4
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        x_t = x.transpose(0, 1)
        att_out, _ = self.attn(x_t, x_t, x_t, key_padding_mask=key_padding_mask)
        x = x + att_out.transpose(0, 1)
        x = self.norm1(x)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class GraphTransformerEncoder(nn.Module):
    def __init__(self, in_dim, d_model=256, n_layers=3):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.layers = nn.ModuleList(
            [SimpleTransformerLayer(d_model) for _ in range(n_layers)]
        )
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, atom_feats, mask):
        x = self.in_proj(atom_feats)
        kpm = ~mask
        for layer in self.layers:
            x = layer(x, key_padding_mask=kpm)
        return self.out_proj(x)


class ResidueAtomCrossAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.a2p = nn.MultiheadAttention(d_model, 4, batch_first=False)
        self.p2a = nn.MultiheadAttention(d_model, 4, batch_first=False)

        self.last_a2p_attn = None
        self.last_p2a_attn = None

    def forward(self, atom_rep, prot_rep, a_mask, p_mask):

        atom_t = atom_rep.transpose(0, 1)
        prot_t = prot_rep.transpose(0, 1)

        a_kpm = ~a_mask
        p_kpm = ~p_mask

        # Save attention maps
        att_a, w_a = self.a2p(atom_t, prot_t, prot_t, key_padding_mask=p_kpm)
        self.last_a2p_attn = w_a.detach().cpu()

        att_p, w_p = self.p2a(prot_t, atom_t, atom_t, key_padding_mask=a_kpm)
        self.last_p2a_attn = w_p.detach().cpu()

        atom_rep = atom_rep + att_a.transpose(0,1)
        prot_rep = prot_rep + att_p.transpose(0,1)

        return atom_rep, prot_rep



class CMGTModel(nn.Module):
    def __init__(self, atom_feat_dim, prot_emb_dim, d_model=256):
        super().__init__()
        self.graph_encoder = GraphTransformerEncoder(atom_feat_dim, d_model)
        self.prot_proj = nn.Linear(prot_emb_dim, d_model)

        self.geo = FourierDistanceEncoding(d_model)
        self.geo_merge = nn.Linear(d_model * 2, d_model)

        self.cross = nn.ModuleList(
            [ResidueAtomCrossAttention(d_model) for _ in range(2)]
        )

        self.pool_fc = nn.Linear(d_model * 2, d_model)
        self.head = nn.Linear(d_model, 1)

    def masked_mean(self, x, mask):
        m = mask.unsqueeze(-1).float()
        return (x * m).sum(1) / m.sum(1).clamp(min=1.0)

    def forward(self, atom_feats, coords, prot_emb, a_mask, p_mask):
        atom_rep = self.graph_encoder(atom_feats, a_mask)
        geo = self.geo(coords, a_mask)
        atom_rep = self.geo_merge(torch.cat([atom_rep, geo], dim=-1))

        prot_rep = self.prot_proj(prot_emb)

        for layer in self.cross:
            atom_rep, prot_rep = layer(atom_rep, prot_rep, a_mask, p_mask)

        atom_pool = self.masked_mean(atom_rep, a_mask)
        prot_pool = self.masked_mean(prot_rep, p_mask)

        joint = torch.cat([atom_pool, prot_pool], dim=-1)
        out = self.head(self.pool_fc(joint))
        return out.squeeze(-1)