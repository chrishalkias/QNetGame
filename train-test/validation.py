from rl_stack import QRNAgent

agent = QRNAgent()

results = agent.validate(
    model_path="checkpoints/policy.pth",
    n_episodes=1,
    max_steps=100,
    n_repeaters=4,           
    p_gen=1, 
    p_swap=1,  
    cutoff=70, 
    F0=1.0, 
    channel_loss=0.0,
    dt_seconds=1e-4,
    plot_actions=1,
    save_dir='assets/',
    ee=True
)