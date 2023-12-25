import torch 
from environment import Env

def hit_metric(recommended, actual):
    return int(actual in recommended)

def dcg_metric(recommended, actual):
    if actual in recommended:
        index = recommended.index(actual)
        return np.reciprocal(np.log2(index + 2))
    return 0

def run_evaluation(net, state_representation, training_env_memory, loader):
    hits = []
    dcgs = []
    test_env = Env(test_matrix)
    test_env.memory = torch.tensor(training_env_memory).cuda()
    user, memory = test_env.reset(int(to_np(next(iter(valid_loader))['user'])[0]))

    for batch in loader:
        batch['user'] = batch['user'].cuda()
        batch['item'] = batch['item'].cuda()
        action_emb = net(state_repr(user, memory))
        scores, action = net.get_action(
            batch['user'], 
            torch.tensor(test_env.memory[to_np(batch['user']).astype(int), :]).cuda(), 
            state_representation.cuda(), 
            action_emb.cuda(),
            batch['item'].long(), 
            return_scores=True
        )
        user, memory, reward, done = test_env.step(action)

        _, ind = scores[:, 0].topk(10)
        predictions = torch.take(batch['item'], ind).cpu().numpy().tolist()
        actual = batch['item'][0].item()
        hits.append(hit_metric(predictions, actual))
        dcgs.append(dcg_metric(predictions, actual))
        
    return np.mean(hits), np.mean(dcgs)