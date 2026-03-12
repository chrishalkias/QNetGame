from rl_stack import QRNAgent

agent = QRNAgent(node_dim=8, #fixed
                 hidden=64,  #fixed
                 lr=5e-4, 
                 gamma=0.99, 
                 buffer_size=10_000,
                 batch_size=64, 
                 tau = 0.005,
                 epsilon=1)



metrics = agent.train(
    episodes=1000,
    max_steps=30,
    n_range=[4],               # re-sample N every episode
    curriculum=True,          # progressive difficulty
    heterogeneous=False,       # randomise per-repeater params
    p_gen=0.8, 
    p_swap=0.85,
    cutoff=15,
    channel_loss=0.0,
    F0=1.0,
    dt_seconds = 0,#1e-4
    save_path='checkpoints/', # "checkpoints/"
    plot=True,
)