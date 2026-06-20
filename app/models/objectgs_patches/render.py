#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# Patched by ToThinkVision:
#   - Fix radii shape mismatch [N, 2] → [N]
#   - Pack semantics into colors (gsplat >= 1.0 removed 'features' kwarg)
#   - Extend backgrounds to match packed channel count
#   - Split packed output into RGB + semantics after rendering

import torch
import math
import gsplat
from gsplat.cuda._wrapper import fully_fused_projection, fully_fused_projection_2dgs


def render(viewpoint_camera, pc, pipe, bg_color, visible_mask=None, training=True, object_mask=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    if pc.explicit_gs:
        xyz, color, opacity, scaling, rot, sh_degree, selection_mask = pc.generate_explicit_gaussians(visible_mask)
        semantics = None
    else:
        if object_mask is None:
            xyz, offset, color, opacity, scaling, rot, sh_degree, selection_mask, semantics = pc.generate_neural_gaussians(viewpoint_camera, visible_mask, training)
        else:
            xyz, offset, color, opacity, scaling, rot, sh_degree, selection_mask, semantics = pc.generate_neural_gaussians(viewpoint_camera, visible_mask & object_mask, training)

    # Set up rasterization configuration
    K = torch.tensor([
            [viewpoint_camera.fx, 0, viewpoint_camera.cx],
            [0, viewpoint_camera.fy, viewpoint_camera.cy],
            [0, 0, 1],
        ],dtype=torch.float32, device="cuda")
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1) # [4, 4]

    # Pack semantics into colors for gsplat compatibility (>=1.0 removed 'features' kwarg)
    if semantics is not None:
        packed_colors = torch.cat([color, semantics.detach()], dim=-1)
        # Extend backgrounds with zeros for semantic channels
        sem_bg = torch.zeros(1, semantics.shape[-1], device=bg_color.device)
        packed_bg = torch.cat([bg_color[None], sem_bg], dim=-1)
    else:
        packed_colors = color
        packed_bg = bg_color[None]

    if pc.gs_attr == "3D":
        _packed_colors, render_alphas, info = gsplat.rasterization(
            means=xyz,  # [N, 3]
            quats=rot,  # [N, 4]
            scales=scaling,  # [N, 3]
            opacities=opacity.squeeze(-1),  # [N,]
            colors=packed_colors,
            viewmats=viewmat[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            backgrounds=packed_bg,
            packed=False,
            sh_degree=sh_degree,
            render_mode=pc.render_mode,
        )
    elif pc.gs_attr == "2D":
        if pc.render_mode in ["RGB+D", "RGB+ED"]:
            packed_bg_2d = torch.cat([packed_bg, torch.zeros(1, 1, device=bg_color.device)], dim=-1)
        else:
            packed_bg_2d = packed_bg

        (_packed_colors,
        render_alphas,
        render_normals,
        render_normals_from_depth,
        render_distort,
        render_median,
        info) = \
        gsplat.rasterization_2dgs(
            means=xyz,  # [N, 3]
            quats=rot,  # [N, 4]
            scales=scaling,  # [N, 3]
            opacities=opacity.squeeze(-1),  # [N,]
            colors=packed_colors,
            viewmats=viewmat[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            backgrounds=packed_bg_2d,
            packed=False,
            sh_degree=sh_degree,
            render_mode=pc.render_mode,
        )
    else:
        raise ValueError(f"Unknown gs_attr: {pc.gs_attr}")

    # Split packed output into RGB + semantics
    if semantics is not None and _packed_colors.shape[-1] > 4:
        sem_dim = semantics.shape[-1]
        render_colors = _packed_colors[..., :-sem_dim]
        render_semantics = _packed_colors[..., -sem_dim:]
    elif semantics is not None and _packed_colors.shape[-1] > 3:
        render_colors = _packed_colors[..., :3]
        render_semantics = _packed_colors[..., 3:]
    else:
        render_colors = _packed_colors
        render_semantics = None

    # [1, H, W, C] -> [C, H, W]
    if render_colors.shape[-1] == 4:
        colors, depths = render_colors[..., 0:3], render_colors[..., 3:4]
        depth = depths[0].permute(2, 0, 1)
    else:
        colors = render_colors
        depth = None

    rendered_image = colors[0].permute(2, 0, 1)
    radii = info["radii"].squeeze(0) # [N,] or [N, 2]
    # Fix: handle radii with multiple channels
    if radii.dim() > 1:
        radii = radii.max(dim=-1).values
    try:
        info["means2d"].retain_grad() # [1, N, 2]
    except:
        pass

    render_alphas = render_alphas[0].permute(2, 0, 1)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    return_dict = {
        "render": rendered_image,
        "scaling": scaling,
        "viewspace_points": info["means2d"],
        "visibility_filter" : radii > 0,
        "visible_mask": visible_mask,
        "selection_mask": selection_mask,
        "opacity": opacity,
        "render_depth": depth,
        "radii": radii,
        "render_alphas": render_alphas,
        "render_semantics": render_semantics,
    }

    if pc.gs_attr == "2D":
        return_dict.update({
            "render_normals": render_normals,
            "render_normals_from_depth": render_normals_from_depth,
            "render_distort": render_distort,
        })

    return return_dict

def prefilter_voxel(viewpoint_camera, pc):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    means = pc.get_anchor[pc._anchor_mask]
    scales = pc.get_scaling[pc._anchor_mask][:, :3]
    quats = pc.get_rotation[pc._anchor_mask]

    # Set up rasterization configuration
    Ks = torch.tensor([
            [viewpoint_camera.fx, 0, viewpoint_camera.cx],
            [0, viewpoint_camera.fy, viewpoint_camera.cy],
            [0, 0, 1],
        ],dtype=torch.float32, device="cuda")[None]
    viewmats = viewpoint_camera.world_view_transform.transpose(0, 1)[None]

    N = means.shape[0]
    C = viewmats.shape[0]
    device = means.device
    assert means.shape == (N, 3), means.shape
    assert quats.shape == (N, 4), quats.shape
    assert scales.shape == (N, 3), scales.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape

    if pc.gs_attr == "3D":
        proj_results = fully_fused_projection(
            means,
            None,
            quats,
            scales,
            viewmats,
            Ks,
            int(viewpoint_camera.image_width),
            int(viewpoint_camera.image_height),
            eps2d=0.3,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            radius_clip=0.0,
            sparse_grad=False,
            calc_compensations=False,
        )
    elif pc.gs_attr == "2D":
        proj_results = fully_fused_projection_2dgs(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            int(viewpoint_camera.image_width),
            int(viewpoint_camera.image_height),
            eps2d=0.3,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            radius_clip=0.0,
            sparse_grad=False,
        )
    else:
        raise ValueError(f"Unknown gs_attr: {pc.gs_attr}")

    radii, means2d, depths, conics, compensations = proj_results
    camera_ids, gaussian_ids = None, None

    visible_mask = pc._anchor_mask.clone()
    # Fix: handle radii with multiple channels
    r = radii.squeeze(0)
    if r.dim() > 1:
        r = r.max(dim=-1).values
    visible_mask[pc._anchor_mask] = r > 0

    return visible_mask
