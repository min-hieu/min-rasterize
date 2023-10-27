import torch
import trimesh
import numpy as np
from PIL import Image

def projection(x=0.1, n=1.0, f=50.0, device='cpu'):
    return torch.tensor([[n/x,    0,            0,              0],
                         [  0,  n/x,            0,              0],
                         [  0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                         [  0,    0,           -1,              0]],
                        dtype=torch.float32, device=device)

def translate(x, y, z, device='cpu'):
    return torch.tensor([[1, 0, 0, x],
                         [0, 1, 0, y],
                         [0, 0, 1, z],
                         [0, 0, 0, 1]],
                        dtype=torch.float32, device=device)

def rot_x(a, device='cpu'):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[1,  0, 0, 0],
                         [0,  c, s, 0],
                         [0, -s, c, 0],
                         [0,  0, 0, 1]],
                        dtype=torch.float32, device=device)

def rot_y(a, device='cpu'):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[ c, 0, s, 0],
                         [ 0, 1, 0, 0],
                         [-s, 0, c, 0],
                         [ 0, 0, 0, 1]],
                        dtype=torch.float32, device=device)

def length(v, dim=-1):
    return (torch.sum(v**2, dim, keepdim=True)+1e-8)**0.5

def normalize(v, dim=-1):
    return v / (torch.sum(v**2, dim, keepdim=True)+1e-8)**0.5

def transform_pos(mtx, pos, device='cpu'):
    t_mtx = torch.from_numpy(mtx).to(device) if isinstance(mtx, np.ndarray) else mtx
    # (x,y,z) -> (x,y,z,1)
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).to(device)], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]

def transform_norm(mtx, norm, device='cpu'):
    norm_shape = norm.shape
    norm = norm.reshape(-1,3)
    t_mtx = torch.from_numpy(mtx).to(device) if isinstance(mtx, np.ndarray) else mtx
    # (x,y,z) -> (x,y,z,0)
    normw = torch.cat([norm, torch.zeros([norm.shape[0], 1]).to(device)], axis=1)
    return torch.matmul(normw, t_mtx.t())[...,:3].reshape(*norm_shape)

def save_img(col):
    col = col[0].detach().cpu().numpy()[::-1]
    img = Image.fromarray(np.clip(np.rint(col*255.0), 0, 255).astype(np.uint8))
    img.save("out.png")

def _to_unit_cube(p):
    mi,ma = p.min(axis=0), p.max(axis=0)
    t, s  = (ma+mi)/2, (ma-mi)/2
    return (p-t)/s

def load_mesh(path, device='cpu', c=0.4):
    mesh = trimesh.load("./bunny.obj")
    v,f,n = _to_unit_cube(mesh.vertices), mesh.faces, mesh.vertex_normals
    v = torch.from_numpy(v.astype(np.float32)).to(device)
    f = torch.from_numpy(f.astype(np.int32)).to(device)
    n = torch.from_numpy(n.astype(np.float32)).to(device)
    c = torch.ones_like(v) * c
    return v,f,n,c
