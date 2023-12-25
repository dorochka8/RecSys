import torch 
from tqdm import tqdm

from agents import State_Repr_Module, Actor_DRR, Critic_DRR
from data.utils import EvalDataset, Prioritized_Buffer, get_beta, preprocess_data
from params import params
from eval import run_evaluation



def ddpg_update(training_env, 
                step=0,
                batch_size=params.batch_size, 
                gamma=params.gamma,
                min_value=params.min_value,
                max_value=params.max_value,
                soft_tau=params.soft_tau,
               ):
    beta = get_beta(step)
    user, memory, action, reward, next_user, next_memory, done = replay_buffer.sample(batch_size, beta)
    user        = torch.FloatTensor(user)
    memory      = torch.FloatTensor(memory)
    action      = torch.FloatTensor(action)
    reward      = torch.FloatTensor(reward)
    next_user   = torch.FloatTensor(next_user)
    next_memory = torch.FloatTensor(next_memory)
    done = torch.FloatTensor(done)
    
    state       = state_repr(user, memory)
    policy_loss = value_net(state, policy_net(state))
    policy_loss = -policy_loss.mean()
    
    next_state     = state_repr(next_user, next_memory)
    next_action    = target_policy_net(next_state)
    target_value   = target_value_net(next_state, next_action.detach())
    expected_value = reward + (1.0 - done) * gamma * target_value
    expected_value = torch.clamp(expected_value, min_value, max_value)

    value = value_net(state, action)
    value_loss = value_criterion(value, expected_value.detach())
    
    state_repr_optimizer.zero_grad()
    policy_optimizer.zero_grad()
    policy_loss.backward(retain_graph=True)
    policy_optimizer.step()

    value_optimizer.zero_grad()
    value_loss.backward(retain_graph=True)
    value_optimizer.step()
    state_repr_optimizer.step()

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                )

    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )




torch.manual_seed(222)

state_repr = State_Repr_Module(user_num, item_num, params.embedding_dim, params.hidden_dim).cuda()
policy_net = Actor_DRR(params.embedding_dim, params.hiddem_dim).cuda()
value_net  = Critic_DRR(params.embedding_dim * 3, params.embedding_dim, params.hiddem_dim).cuda()
replay_buffer = Prioritized_Buffer(params.buffer_size)

target_value_net  = Critic_DRR(params.embedding_dim * 3, params.embedding_dim, params.hiddem_dim)
target_policy_net = Actor_DRR(params.embedding_dim, params.hidden_dim)

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)

for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
    target_param.data.copy_(param.data)

value_criterion  = nn.MSELoss()
value_optimizer  = Ranger(value_net.parameters(),  lr=params.value_lr, 
                          weight_decay=params.value_decay)
policy_optimizer = Ranger(policy_net.parameters(), lr=params.policy_lr, 
                          weight_decay=params.policy_decay)
state_repr_optimizer = Ranger(state_repr.parameters(), lr=params.state_repr_lr, 
                              weight_decay=params.state_repr_decay)

writer = SummaryWriter(log_dir=params.log_dir)

np.random.seed(16)
train_env = Env(train_matrix)
hits, dcgs = [], []
hits_all, dcgs_all = [], []
step, best_step = 0, 0
step, best_step, best_step_all = 0, 0, 0
users = np.random.permutation(appropriate_users)

if not os.path.isdir('./data'):
    os.mkdir('./data')
    
file_path = os.path.join(data_dir, rating)
if os.path.exists(file_path):
    print("Skip loading " + file_path)
else:
    with open(file_path, "wb") as tf:
        print("Load " + file_path)
        r = requests.get("https://raw.githubusercontent.com/hexiangnan/neural_collaborative_filtering/master/Data/" + rating)
        tf.write(r.content)
        
(train_data, train_matrix, test_data, test_matrix, 
 user_num, item_num, appropriate_users) = preprocess_data(data_dir, rating)

for u in tqdm.tqdm(users):
    user, memory = train_env.reset(u)
    # user, memory = user.to('cuda'), memory.to('cuda')
    for t in range(int(train_matrix[u].sum())):
        action_emb = policy_net(state_repr(user, memory))
        action = policy_net.get_action(
            user, 
            torch.tensor(train_env.memory[to_np(user).astype(int), :]).cuda(), 
            state_repr, 
            action_emb,
            torch.tensor(
                [item for item in train_env.available_items 
                if item not in train_env.viewed_items]
            ).long().cuda()
        )
        user, memory, reward, done = train_env.step(
            action, 
            action_emb,
            buffer=replay_buffer
        )

        if len(replay_buffer) > params.batch_size:
            ddpg_update(train_env, step=step)

        if (step+1) % 100 == 0 and step > 0:
            hit, dcg = run_evaluation(policy_net, state_repr, torch.FloatTensor(train_env.memory).cuda())
            writer.add_scalar('hit', hit, step)
            writer.add_scalar('dcg', dcg, step)
            hits.append(hit)
            dcgs.append(dcg)
            if np.mean(np.array([hit, dcg]) - np.array([hits[best_step], dcgs[best_step]])) > 0:
                best_step = step // 100
                torch.save(policy_net.state_dict(), params.log_dir + 'policy_net.pth')
                torch.save(value_net.state_dict(), params.log_dir + 'value_net.pth')
                torch.save(state_repr.state_dict(), params.log_dir + 'state_repr.pth')
        if step % 10000 == 0 and step > 0:
            hit, dcg = run_evaluation(policy_net, state_repr, train_env.memory, full_loader)
            writer.add_scalar('hit_all', hit, step)
            writer.add_scalar('dcg_all', dcg, step)
            hits_all.append(hit)
            dcgs_all.append(dcg)
            if np.mean(np.array([hit, dcg]) - np.array([hits_all[best_step_all], dcgs_all[best_step_all]])) > 0:
                best_step_all = step // 10000
                torch.save(policy_net.state_dict(), params.log_dir + 'best_policy_net.pth')
                torch.save(value_net.state_dict(), params.log_dir + 'best_value_net.pth')
                torch.save(state_repr.state_dict(), params.log_dir + 'best_state_repr.pth')
        step += 1

torch.save(policy_net.state_dict(), params.log_dir + 'policy_net_final.pth')
torch.save(value_net.state_dict(), params.log_dir + 'value_net_final.pth')
torch.save(state_repr.state_dict(), params.log_dir + 'state_repr_final.pth')
with open('logs/memory.pickle', 'wb') as f:
    pickle.dump(train_env.memory, f)
    
with open('logs/memory.pickle', 'rb') as f:
    memory = pickle.load(f)

state_repr = State_Repr_Module(user_num, item_num, params.embedding_dim, params.hidden_dim)
policy_net = Actor_DRR(params.embedding_dim, params.hiddem_dim)
state_repr.load_state_dict(torch.load('logs/' + 'best_state_repr.pth'))
policy_net.load_state_dict(torch.load('logs/' + 'best_policy_net.pth'))
    
hit, dcg = run_evaluation(policy_net, state_repr, memory, full_loader)
print('hit rate: ', hit, 'dcg: ', dcg)