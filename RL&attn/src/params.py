class params:
    batch_size = 512
    embedding_dim = 8
    hidden_dim = 16
    N = 5  
    value_lr = 1e-5
    value_decay = 1e-4
    policy_lr = 1e-5
    policy_decay = 1e-6
    state_repr_lr = 1e-5
    state_repr_decay = 1e-3
    log_dir = 'logs/'
    gamma = 0.8
    min_value = -10
    max_value = 10
    soft_tau = 1e-3
    buffer_size = 1000000
    user_num = 