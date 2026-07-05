"""TEMP: fine-tune the omni_nopen_15k agent in an environment WITH classical-
communication (CC) delays (1 step/hop), reusing the full QRNAgent.train pipeline
(curriculum-over-N, per-repeater inhomogeneities sigma=0.15, winnability oracle)
and only adding the fine-tuning knobs now supported by train():

  * warm-start: load the 15k weights into BOTH policy and target nets.
  * smaller LR (1e-4 vs 5e-4 pretrain) + exponential decay (lr_decay).
  * LOW initial epsilon (eps_init=0.3, cosine-annealed to 0.05) -- do NOT restart
    exploration at 1.0, which would scramble the pretrained policy; we only want
    to explore the NEW CC-delay dynamics.

CC timing: dt_seconds = spacing / c_fiber => a k-hop event resolves after k steps
(1 step/hop). channel_loss=0 so distance affects only the delay.

  PYTHONPATH=. python experiments/temp/finetune_ccdelay.py --episodes 2000
"""
from __future__ import annotations
import argparse, os
import torch
from rl_stack import QRNAgent

C_FIBER_KM_S = 200_000.0   # matches RepeaterNetwork.c_fiber


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="checkpoints/omni_initial/omni_nopen_15k/policy.pth")
    ap.add_argument("--run_id", default="ft_ccdelay_2k")
    ap.add_argument("--save_base_dir", default="checkpoints")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=300,
                    help="larger than the no-CC 200 since CC delays slow delivery")
    # fine-tuning knobs
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_decay", type=float, default=0.999, help="ExponentialLR gamma/episode")
    ap.add_argument("--eps_init", type=float, default=0.30)
    ap.add_argument("--eps_fin", type=float, default=0.05)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--gamma", type=float, default=0.995)
    # curriculum / system (mirrors submit_omni_15k; N ceiling lowered to 10 so
    # CC-delayed episodes stay deliverable inside max_steps)
    ap.add_argument("--n_lo", type=int, default=4)
    ap.add_argument("--n_hi", type=int, default=10)
    ap.add_argument("--n_ch", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--p_gen", type=float, nargs=2, default=[0.4, 0.9])
    ap.add_argument("--p_swap", type=float, nargs=2, default=[0.4, 0.9])
    ap.add_argument("--p_gen_std", type=float, default=0.15)
    ap.add_argument("--p_swap_std", type=float, default=0.15)
    ap.add_argument("--cutoff_lo", type=int, default=10)
    ap.add_argument("--cutoff_hi", type=int, default=40)
    ap.add_argument("--spacing", type=float, default=50.0)
    ap.add_argument("--dt_seconds", type=float, default=-1.0,
                    help="<0 -> spacing/c_fiber (1 step/hop)")
    return ap.parse_args()


def main():
    a = parse_args()
    dt = a.spacing / C_FIBER_KM_S if a.dt_seconds < 0 else a.dt_seconds
    sph = a.spacing / (C_FIBER_KM_S * dt) if dt > 0 else 0.0
    save_path = os.path.join(a.save_base_dir, a.run_id)
    os.makedirs(save_path, exist_ok=True)

    agent = QRNAgent(lr=a.lr, hidden=a.hidden, batch_size=a.batch_size,
                     buffer_size=80_000, gamma=a.gamma, tau=0.005,
                     epsilon=a.eps_init)

    # warm-start policy AND target from the pretrained checkpoint
    sd = torch.load(a.ckpt, map_location=agent.device)
    agent.policy_net.load_state_dict(sd)
    agent.target_net.load_state_dict(sd)
    print(f"warm-started from {a.ckpt}  |  dt={dt:.3e} ({sph:.2f} step/hop)  "
          f"lr={a.lr} decay={a.lr_decay}  eps {a.eps_init}->{a.eps_fin}", flush=True)

    agent.train(
        episodes=a.episodes,
        max_steps=a.max_steps,
        n_range=list(range(a.n_lo, a.n_hi + 1)),
        curriculum=True,
        n_ch=a.n_ch,
        p_gen=tuple(a.p_gen),
        p_swap=tuple(a.p_swap),
        p_gen_std=a.p_gen_std,
        p_swap_std=a.p_swap_std,
        cutoff=(a.cutoff_lo, a.cutoff_hi),
        channel_loss=0.0,
        F0=1.0,
        dt_seconds=dt,
        topology="chain",
        prune_unwinnable=True,
        save_path=save_path,
        save_best=True,
        eps_init=a.eps_init,
        eps_fin=a.eps_fin,
        lr_decay=a.lr_decay,
        plot=True,
    )
    print(f"done -> {save_path}/policy.pth", flush=True)


if __name__ == "__main__":
    main()
