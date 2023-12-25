import numpy as np
import torch 

from params import params


class Env:
    def __init__(self, user_item_matrix, user_num):
        self.matrix = user_item_matrix
        self.item_count = item_num
        self.memory = np.full((user_num, params.N), item_num)

    def reset(self, user_id):
        self.user_id = user_id
        self.viewed_items = []
        self.related_items = np.argwhere(self.matrix[self.user_id] > 0)[:, 1]
        nonrelated_items = set(range(self.item_count)) - set(self.related_items)
        self.nonrelated_items = np.random.choice(list(nonrelated_items), len(self.related_items))
        
        self.available_items = np.empty(len(self.related_items) * 2)
        self.available_items[::2] = self.related_items
        self.available_items[1::2] = self.nonrelated_items

        user_tensor = torch.tensor([self.user_id]).cuda()
        memory_tensor = torch.tensor(self.memory[[self.user_id], :]).cuda()
        return user_tensor, memory_tensor

    def step(self, action, action_emb=None, buffer=None):
        initial_user = self.user_id
        initial_memory = self.memory[[initial_user], :]

        reward = float(action.detach().cpu().numpy()[0] in self.related_items)
        self.viewed_items.append(action.detach().cpu().numpy()[0])
        action_value = action.cpu().item() if len(action) == 1 else action.cpu()[0].item()
        self.memory[self.user_id] = np.roll(self.memory[self.user_id], -1)
        self.memory[self.user_id][-1] = action_value

        done = int(len(self.viewed_items) == len(self.related_items))

        if buffer:
            buffer.push(np.array([initial_user]), np.array(initial_memory), 
                        action_emb.detach().cpu().numpy()[0], np.array([reward]), 
                        np.array([self.user_id]), self.memory[[self.user_id], :], 
                        np.array([reward]))

        user_tensor = torch.tensor([self.user_id]).cuda()
        memory_tensor = torch.tensor(self.memory[[self.user_id], :]).cuda()
        return user_tensor, memory_tensor, reward, done
