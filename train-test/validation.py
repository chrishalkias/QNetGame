from rl_stack import QRNAgent

agent = QRNAgent()

results = agent.validate(
    model_path="checkpoints/policy.pth",
    n_episodes=1,
    max_steps=70,
    n_repeaters=4,           
    p_gen=0.2, 
    p_swap=0.8,  
    cutoff=25, 
    F0=1.0, 
    channel_loss=0.00,
    dt_seconds=0, #1e-4
    plot_actions=True,
    save_dir='assets/',
    ee=True
)