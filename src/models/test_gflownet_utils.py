# %%
import torch
from collections import OrderedDict
from src.models.gflownet_utils import (
    Vocab,
    FlowGenerator,
    compute_reward_distribution,
    StandardDataset
)
import matplotlib.pyplot as pp
import numpy as np
# from maia.algorithms.utils_train import StandardDataset

# Get initial state
def reward_func(protocol):

    state_1 = torch.tensor([[0, 0], [1.5, 0.5], [1.5, 0.5]])
    state_2 = torch.tensor([[0, 0], [0, 0], [0, 0]])
    state_3 = torch.tensor([[0, 0], [0.5, 0.5], [0.5, 0.5]])

    match_1 = torch.all(protocol == state_1).item()
    match_2 = torch.all(protocol == state_2).item()
    match_3 = torch.all(protocol == state_3).item()

    if match_1:
        # print("Matched best state!")
        return 1

    elif match_2:
        # print("Matched second best state!")
        return 0.5

    elif match_3:
        # print("Matched third best state!")
        return 0.25

    else:
        return 1e-4


# %%
# Running simplest test example
alphabet_set = OrderedDict(
    {
        "ox": [0.0, 0.5, 1.5],
        "ph": [0.0, 0.5],
    }
)

# %%
vocab = Vocab(alphabet=alphabet_set)

# %%
vocab.alphabet

# %%
seq_len = 3
num_pressures = len(alphabet_set)
num_tokens = vocab.num_tokens
init_protocol = OrderedDict({"ox": 0.0, "ph": 0.0})  # , "stir": 0.0, "temp": 0.0})
reward_distribution, empirical_reward_distribution, dataset_np, tensor_to_key = compute_reward_distribution(
    alphabet_set, reward_func, seq_len, init_protocol
)

gt = np.array(list(reward_distribution.values()))

index = len(dataset_np) // 2
partial_dataset = StandardDataset(torch.from_numpy(dataset_np[:index]).float(), torch.from_numpy(gt[:index]).float())
full_dataset = StandardDataset(torch.from_numpy(dataset_np).float(), torch.from_numpy(gt).float())

# %%
gflownet = FlowGenerator(alphabet_set, seq_len, delta=0.1, test=True, verbose=False)

# %%
tb_losses, tb_mean_loss, logZs = gflownet.train(partial_dataset.x, reward_func, num_episodes=10000)

# %%
# # %%
# # Continuing training
# tb_losses, logZs, sum_rewards = gflownet.train(
#     partial_dataset.x, reward_func, num_episodes=50000, tb_losses=tb_losses, logZs=logZs
# )

# %%
sum_rewards = np.sum([reward_func(protocol) for protocol in full_dataset.x])

# %%
sum_rewards

# %%
f, ax = pp.subplots(3, 1, figsize=(10, 6))
pp.sca(ax[0])
pp.plot(tb_losses)
pp.yscale("log")
pp.ylabel("loss")

pp.sca(ax[1])
pp.plot(np.exp(logZs))
pp.axhline(sum_rewards, color="red", linestyle="--")
pp.ylabel("estimated Z")
# pp.ylim(0, sum_rewards + 50)

pp.sca(ax[2])
pp.plot(tb_mean_loss)
pp.yscale("log")
pp.ylabel("mean loss")


# %%
np.exp(logZs)[-500:]

# %%
# Sample from GFlowNet
tb_sampled_protocols = gflownet.sample(sample_len=1000)

# %%
for gen_protocol in tb_sampled_protocols:
    if str(gen_protocol) in tensor_to_key:
        key = tensor_to_key[str(gen_protocol)]
        empirical_reward_distribution[key] += 1
    else:
        print("Outside protocol!")


# %%
pp.bar(
    list(empirical_reward_distribution.keys()),
    np.array(list(empirical_reward_distribution.values())) / np.sum(list(empirical_reward_distribution.values())),
    label="Empirical",
    alpha=0.5,
)
pp.bar(
    list(reward_distribution.keys()),
    np.array(list(reward_distribution.values())) / np.sum(list(reward_distribution.values())),
    label="Expected",
    alpha=0.5,
)

pp.xticks(list(empirical_reward_distribution.keys()), list(empirical_reward_distribution.keys()), rotation="vertical")
pp.legend()

# %%
