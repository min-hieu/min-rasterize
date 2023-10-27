import torch
import nvdiffrast.torch as dr

from render_utils import *

@torch.no_grad()
def render(
    mvp,
    pos, pos_idx,
    col, col_idx=None,
    res=64, device='cpu',
):
    glcrx   = dr.RasterizeCudaContext(device)
    col_idx = pos_idx if col_idx is None else col_idx

    pos_clip    = transform_pos(mvp, pos, device)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[res, res], grad_db=False)
    color, _    = dr.interpolate(col[None, ...], rast_out, col_idx)
    color       = dr.antialias(color, rast_out, pos_clip, pos_idx)

    return color

@torch.no_grad()
def render_diffuse(
    mvp, nm, vdir,
    pos, pos_idx, normals,
    col, col_idx=None,
    res=64, device='cpu',
    ldir=None, bg_color=0.2,
):
    glctx   = dr.RasterizeCudaContext(device)
    ldir    = vdir if ldir is None else ldir
    zero    = torch.as_tensor(0.0, dtype=torch.float32, device=device)
    col_idx = pos_idx if col_idx is None else col_idx

    Kd = torch.tensor([0.1, 0.1, 0.1], device=device) # diffuse
    Ka = torch.tensor([0.2, 0.2, 0.2], device=device) # ambient

    viewvec = pos[..., :3] - vdir[None, None, :] # View vectors at vertices.
    reflvec = viewvec - 2.0 * normals[None, ...] * torch.sum(normals[None, ...] * viewvec, -1, keepdim=True) # Reflection vectors at vertices.
    reflvec = reflvec / torch.sum(reflvec**2, -1, keepdim=True)**0.5 # Normalize.

    # Rasterize
    pos_clip    = transform_pos(mvp, pos, device)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, [res, res], grad_db=False)

    normals  = normalize(normals).contiguous()
    norms, _ = dr.interpolate(normals, rast_out, pos_idx)
    norms    = normalize(transform_norm(nm, norms, device))

    # Phong shading
    ndotl = torch.sum(-ldir * norms, -1, keepdim=True)
    color = Ka + Kd * ndotl.clip(min=0)
    color = torch.where(rast_out[..., -1:] == 0, bg_color, color)
    color = dr.antialias(color, rast_out, pos_clip, pos_idx)

    return color

def main():
    device  = 1
    v,f,n,c = load_mesh("./bunny.obj", device)

    rot = rot_x(-0.4) @ rot_y(0.0)
    t   = torch.tensor([0,0,-3.5], dtype=torch.float32, device=device)
    proj = projection(x=0.7, device=device)
    mv   = (translate(*t) @ rot).to(device)
    mvp  = proj @ mv
    nm   = mv.inverse().T # normal matrix

    color = render_diffuse(mvp, nm, t, v, f, n, c, res=128, device=device)
    save_img(color)

if __name__ == '__main__':
    main()
