import torch
import numpy as np
class SampleDataset(torch.utils.data.Dataset):
    def __init__(self, sample):
        super(SampleDataset, self).__init__()
        self.sample_list = sample
      
    def __getitem__(self, idx):
        
    
        sample = self.sample_list[idx]
        return sample

    def update(self, sample):
        self.sample_list = torch.cat([self.sample_list, sample], dim=0)

    def deque(self, length):
        self.sample_list = self.sample_list[length:]

    def get_seq(self):
        return self.sample_list

    def __len__(self):
        return len(self.sample_list)

    def collate(data_list):
        return torch.stack(data_list)

class RewardDataset(torch.utils.data.Dataset):
    def __init__(self, rewards):
        super(RewardDataset, self).__init__()
        self.rewards = rewards
        self.raw_tsrs = self.rewards

    def __getitem__(self, idx):
        return self.rewards[idx]
        #return  self.score_list[idx]


    def update(self, rewards):
        new_rewards = rewards

        self.raw_tsrs = torch.cat([self.rewards, new_rewards], dim=0)
        self.rewards = self.raw_tsrs

    def deque(self, length):
        self.raw_tsrs = self.raw_tsrs[length:]
        self.rewards = self.raw_tsrs


    def get_tsrs(self):
        return self.rewards

    def __len__(self):
        return self.rewards.size(0)

    def collate(data_list):
        return torch.stack(data_list)

class ZipDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        return [dataset[idx] for dataset in self.datasets]

    def collate(data_list):
        return [dataset.collate(data_list) for dataset, data_list in zip(self.datasets, zip(*data_list))]


def collate(data_list):
    sample,rewards  = zip(*data_list)

    sample_data = SampleDataset.collate(sample)
    reward_data = RewardDataset.collate(rewards)

    return sample_data, reward_data

    

class ReplayBuffer():
    def __init__(self, buffer_size, device, log_reward, batch_size, data_ndim=2, beta=1.0, rank_weight=1e-2, prioritized=None):
        self.buffer_size = buffer_size
        self.prioritized = prioritized
        self.device = device
        self.data_ndim = data_ndim
        self.batch_size = batch_size
        self.reward_dataset = None
        self.buffer_idx = 0
        self.buffer_full = False
        self.log_reward = log_reward
        self.beta = beta
        self.rank_weight = rank_weight
        self.beta = beta
    def add(self, samples,log_r):
        if self.reward_dataset is None:
            self.reward_dataset = RewardDataset(log_r.detach())
            self.sample_dataset = SampleDataset(samples.detach())
            self.sample_dataset.update(samples.detach())
            self.reward_dataset.update(log_r.detach())
        else:
            self.sample_dataset.update(samples.detach())
            self.reward_dataset.update(log_r.detach())


        
        if self.reward_dataset.__len__() > self.buffer_size:
            self.reward_dataset.deque(self.reward_dataset.__len__() - self.buffer_size)
            self.sample_dataset.deque(self.sample_dataset.__len__() - self.buffer_size)
        if self.prioritized == 'rank':
            self.scores_np = self.reward_dataset.get_tsrs().detach().cpu().view(-1).numpy()
            ranks = np.argsort(np.argsort(-1 * self.scores_np))
            weights = 1.0 / (1e-2 * len(self.scores_np) + ranks)
            self.dataset = ZipDataset(self.sample_dataset,self.reward_dataset)
            self.sampler = torch.utils.data.WeightedRandomSampler(
                    weights=weights, num_samples=len(self.scores_np), replacement=True
                    )

            self.loader = torch.utils.data.DataLoader(
                self.dataset, 
                sampler=self.sampler, 
                batch_size=self.batch_size, 
                collate_fn=collate,
                drop_last=True
                )
        else:   
            weights = 1.0
            self.dataset = ZipDataset(self.sample_dataset,self.reward_dataset)
            self.sampler = torch.utils.data.WeightedRandomSampler(
                    weights=weights, num_samples=len(self.scores_np), replacement=True
                    )

            self.loader = torch.utils.data.DataLoader(
                self.dataset, 
                sampler=self.sampler, 
                batch_size=self.batch_size, 
                collate_fn=collate,
                drop_last=True
                )
        # check if we have any additional samples before updating the buffer and the scorer!


    def sample(self):

        try:
            sample, reward = next(self.data_iter)
        except:
            self.data_iter = iter(self.loader)
            sample, reward = next(self.data_iter)
            
        return sample.detach(), reward.detach()