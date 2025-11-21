# Algorithm: DPS-Vis Guided 3DGS Reconstruction
"""
Inputs:
  I_refs = {(I_i, P_i)}_{i=1..N}   # reference images + cameras
  init_3DGS = build_3dgs(I_refs)   # initialize G_0
  candidate_views = sample_candidate_views()  # sphere sampling or target trajectory
  K = max_outer_iters
  M = num_hard_views_per_round
  S = samples_per_view (e.g., 4)
  T = diffusion_steps
  hyperparams = {alpha,beta,gamma,lambda_dc,lambda_pseudo,...}
"""
G = init_3DGS
for k in range(K):
    # 1. render candidate views and compute difficulty
    diffs = {}
    for v in candidate_views:
        I_gs, D_gs, Vis = render(G, v)
        diffs[v] = compute_diff(I_gs, D_gs, Vis, hyperparams)
    hard_views = top_M(diffs, M)

    # 2. for each hard view, run DPS-Vis sampling to get pseudo GT
    pseudo_set = {}
    for v in hard_views:
        best_sample = None
        best_score = +inf
        for s in range(S):
            x_T = sample_noise()
            for t in reversed(range(1, T+1)):
                # standard denoise step (model conditioned on refs and camera token)
                x_tminus1 = denoise_step(x_T, t, cond=I_refs, cam=v)
                # compute vis-weighted photo gradient (use current G to compute Vis)
                Vis = render_visibility(G, v)
                grad = grad_photo(x_tminus1, I_refs, Vis)
                # apply DPS-Vis update
                x_tminus1 = x_tminus1 + eta_t * lambda_dc * (Vis * grad)
                x_T = x_tminus1
            score = photo_loss(x_T, I_refs, Vis)
            if score < best_score:
                best_score = score; best_sample = x_T
        pseudo_set[v] = best_sample

    # 3. update 3DGS using pseudo GTs + refs
    loss = 0
    for (I_i,P_i) in I_refs:
        loss += photo_loss(render(G,P_i), I_i, render_visibility(G,P_i))
    for v, I_pseudo in pseudo_set.items():
        loss += lambda_pseudo * photo_loss(render(G,v), I_pseudo, render_visibility(G,v))
    loss += lambda_reg * reg(G)
    G = optimize_G_step(G, loss)   # a few gradient steps (or LBFGS/Adam)

    # optional: check convergence criteria -> break

Return G