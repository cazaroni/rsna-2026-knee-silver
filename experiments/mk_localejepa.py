"""ssl_localejepa.py = ssl_lejepa3.py with SIGReg replaced by LocaLeReg (our reading of the LocaLeJEPA post):
split the batch into K neighbourhoods (k-means on the views' mean projection), centre each on its own mean, and match
each to a uniform-in-ball target with identity covariance (radius sqrt(D+2)) by the same Epps-Pulley test on random
1-D projections. The target's 1-D characteristic function is computed once by Monte Carlo."""
src = open("ssl_lejepa3.py").read()
start = src.index("class SIGReg(nn.Module):"); end = src.index("sigreg = SIGReg().to(dev)")
new = '''class LocaLeReg(nn.Module):
    """Local Epps-Pulley test of a max-entropy (uniform-ball, identity-covariance) target per neighbourhood."""
    def __init__(s, dim, k=8, knots=17, slices=256, iters=5):
        super().__init__()
        t = torch.linspace(0, 3, knots); dt = 3 / (knots - 1)
        wts = torch.full((knots,), 2 * dt); wts[[0, -1]] = dt
        g = torch.Generator().manual_seed(0); n = 200000
        u = torch.randn(n, dim, generator=g); u = u / u.norm(dim=1, keepdim=True)
        r = (dim + 2) ** 0.5 * torch.rand(n, 1, generator=g).pow(1.0 / dim)
        x1 = (u * r)[:, 0]                                    # 1-D marginal of the uniform ball (isotropic: any direction)
        phi = torch.cos(x1.unsqueeze(1) * t).mean(0)          # target CF (real; the sine part is 0 by symmetry)
        s.k, s.iters, s.slices = k, iters, slices
        s.register_buffer("t", t); s.register_buffer("phi", phi); s.register_buffer("w", wts * torch.exp(-t.square() / 2))
    @torch.no_grad()
    def groups(s, m):                                         # k-means on the batch (detached)
        c = m[torch.randperm(m.size(0), device=m.device)[: s.k]].clone()
        for _ in range(s.iters):
            a = torch.cdist(m, c).argmin(1)
            for j in range(s.k):
                if (a == j).any(): c[j] = m[a == j].mean(0)
        return torch.cdist(m, c).argmin(1)
    def forward(s, p):                                        # p: (V, B, D)
        a = s.groups(p.detach().mean(0))
        A = torch.randn(p.size(-1), s.slices, device=p.device); A = A / A.norm(dim=0)
        tot, n = 0.0, 0
        for j in range(s.k):
            idx = (a == j).nonzero().squeeze(1)
            if idx.numel() < 4: continue
            q = p[:, idx]; q = q - q.mean(dim=(0, 1), keepdim=True)     # centre the neighbourhood on its own mean
            xt = (q @ A).unsqueeze(-1) * s.t                            # (V, b, slices, knots)
            err = (xt.cos().mean(-3) - s.phi).square() + xt.sin().mean(-3).square()
            tot = tot + (err @ s.w).mean() * idx.numel() * idx.numel(); n += idx.numel()
        return tot / max(n, 1)

'''
src = src[:start] + new + src[end:]
src = src.replace("sigreg = SIGReg().to(dev)", "sigreg = LocaLeReg(a.proj_dim, k=a.k_local).to(dev)")
src = src.replace('ap.add_argument("--lamb", type=float, default=0.02)', 'ap.add_argument("--lamb", type=float, default=0.02)\nap.add_argument("--k_local", type=int, default=8)')
assert "LocaLeReg(a.proj_dim" in src and "--k_local" in src and "class SIGReg" not in src
open("ssl_localejepa.py", "w").write(src); print("written ssl_localejepa.py")
