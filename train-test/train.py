from rl_stack import QRNAgent

agent = QRNAgent(node_dim=7, #fixed
                 hidden=64,  #fixed
                 lr=5e-4, 
                 gamma=0.99, 
                 buffer_size=50000,
                 batch_size=64, 
                 tau = 0.005,
                 epsilon=1)



metrics = agent.train(
    episodes=800,
    max_steps=10,
    n_range=[4],     # train on small chains
    jitter=False,                 # re-sample N every episode
    curriculum=True,          # progressive difficulty
    heterogeneous=False,       # randomise per-repeater params
    p_gen=0.8, 
    p_swap=0.85,
    cutoff=25,
    channel_loss=0.01,
    F0=0.99,
    dt_seconds = 1e-4,
    save_path='checkpoints/', # "checkpoints/"
    plot=True,
)