from __future__ import annotations

from typing import List
import numpy as np
from typing import Dict, List, Mapping, Callable, Tuple

import torch
import torch.nn as nn
import tqdm
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from sklearn.model_selection import ParameterGrid
from itertools import product
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset

class StandardDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        super().__init__()
        # Data Loading
        self.x = x
        self.y = y

        self.n_samples = self.x.shape[0]

    def __getitem__(self, index: int) -> Tuple:
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


# Making tokenizer
class Vocab(object):
    def __init__(
        self,
        alphabet: Mapping[str, List],
    ) -> None:
        self.alphabet = alphabet
        self.max_alphabet_len = 0
        self.alphabet_to_token, self.token_to_alphabet, self.alphabet_index, self.index_alphabet = self.build_vocab()
        self.mask_alpha = self.build_mask(self.alphabet_index)
        self.num_tokens = self.max_alphabet_len

    def build_vocab(self):
        # Build vocabulary and find max alphabet length
        alphabet_to_token = {}
        token_to_alphabet = {}
        alphabet_index = {}
        index_alphabet = {}

        for index, key in enumerate(self.alphabet):
            alphabet_to_i = {}
            i_to_alphabet = {}

            for i, alphabet in enumerate(self.alphabet[key]):
                alphabet_to_i[alphabet] = i
                i_to_alphabet[i] = alphabet

            alphabet_to_token[key] = alphabet_to_i
            token_to_alphabet[key] = i_to_alphabet
            alphabet_index[key] = index
            index_alphabet[index] = key

            self.max_alphabet_len = max(self.max_alphabet_len, len(self.alphabet[key]))

        return alphabet_to_token, token_to_alphabet, alphabet_index, index_alphabet

    def build_mask(self, alphabet_index):
        # Build mask
        mask_alpha = {}
        
        for key in self.alphabet:
            mask_alphabet = torch.zeros(self.max_alphabet_len)
            mask_alphabet[: len(self.alphabet[key])] = 1
            mask_alpha[alphabet_index[key]] = mask_alphabet

        return mask_alpha


def state_to_protocol(one_hot: torch.Tensor, vocab: Vocab) -> np.ndarray:
    protocol = []
    state_max_arg = torch.argmax(one_hot, dim=2).numpy()

    for alphabet_t in state_max_arg:
        alphabet = []
        for i, token in enumerate(alphabet_t):
            alphabet.append(vocab.token_to_alphabet[vocab.index_alphabet[i]][token])
        protocol.append(alphabet)

    return np.array(protocol)


def tokenize_protocol(protocol: List[Dict[str, float]], vocab: Vocab) -> List[Dict[str, int]]:
    tokenized_protocol = []

    for alphabet_t in protocol:
        tokenized_alphabet = []
        for key in alphabet_t.keys():
            tokenized_alphabet.append(vocab.alphabet_to_token[key][alphabet_t[key]])
        tokenized_protocol.append(tokenized_alphabet)

    return tokenized_protocol


def tokenize_tensor(tensor: torch.Tensor, vocab: Vocab) -> torch.Tensor:
    tokenized_tensor = torch.zeros(tensor.shape[0], tensor.shape[1], dtype=int)

    # print("tensor = " + str(tensor))
    for i, pressures_t in enumerate(tensor):
        for j, pressure in enumerate(pressures_t):
            projected_pr_list = np.array(vocab.alphabet[vocab.index_alphabet[j]])
            distances = np.abs(projected_pr_list - pressure.item())
            min_arg = np.argmin(distances)
            projection = projected_pr_list[min_arg]
            tokenized_tensor[i, j] = vocab.alphabet_to_token[vocab.index_alphabet[j]][projection]

    # print("tokenized tensor = " + str(tokenized_tensor))
    return tokenized_tensor


def to_numpy(protocol_dict: List[Dict[str, int]], alphabet_index) -> np.ndarray:
    protocol_np = np.full((len(protocol_dict), len(protocol_dict[0])), np.nan)

    for t in range(len(protocol_dict)):
        alphabets = protocol_dict[t]
        for key in alphabets.keys():
            protocol_np[t, alphabet_index[key]] = alphabets[key]

    return protocol_np


def get_possible_protocols(alphabets: Mapping[str, List], seq_len: int, init_protocol: OrderedDict):
    param_grid = {key: alphabets[key] for key in alphabets}
    possible_sets = list(ParameterGrid(param_grid))

    full_grid = list(product(possible_sets, repeat=seq_len))
    possible_protocols = [protocol for protocol in full_grid if protocol[0] == init_protocol]

    return possible_protocols


def compute_reward_distribution(
    alphabets: Mapping[str, List],
    reward_func: Callable,
    seq_len: int,
    init_protocol: OrderedDict,
    # vocab: Vocab,
):
    # Get possible protocols
    possible_protocols = get_possible_protocols(alphabets, seq_len, init_protocol)

    # Compute expected reward distribution
    reward_distribution = {}
    empirical_reward_distribution = {}
    tensor_to_key = {}
    dataset = []

    for i, protocol in enumerate(possible_protocols):
        protocol_to_numpy = np.array([np.array(list(alphabet.values())) for alphabet in protocol])
        protocol_tensor = torch.from_numpy(protocol_to_numpy).float()
        dataset.append(protocol_to_numpy)
        tensor_to_key[str(protocol_to_numpy)] = i
        reward_distribution[i] = reward_func(protocol_tensor)
        empirical_reward_distribution[i] = 0

    return reward_distribution, empirical_reward_distribution, np.array(dataset), tensor_to_key


class TBModelClosedLoop(nn.Module):
    def __init__(self, vocab: Vocab, seq_len: int, num_pressures: int, num_tokens: int, num_hid: int):
        super().__init__()
        # The input dimension is seq_len * num_pressures * num_tokens.
        self.vocab = vocab
        self.seq_len = seq_len
        self.num_pressures = num_pressures
        self.num_tokens = num_tokens

        self.input_dim = self.seq_len * self.num_pressures * self.num_tokens
        self.output_dim = self.seq_len * self.num_pressures * self.num_tokens

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, num_hid),
            nn.LeakyReLU(),
            nn.Linear(num_hid, self.output_dim),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # for key in self.vocab.mask_pr.keys():
        #     self.vocab.mask_pr[key] = self.vocab.mask_pr[key].to(self.device)

        # logZ is just a single number
        self.logZ_nn = nn.Parameter(torch.ones(1))

    @property
    def logZ(self):
        return self.logZ_nn

    def forward(self, state, t, pr):

        input_mlp = state.view(-1, self.seq_len * self.num_pressures * self.num_tokens)
        logits = self.mlp(input_mlp)

        # Slice the logits, and mask invalid actions (since we're predicting
        # log-values), we use -100 since exp(-100) is tiny, but we don't want -inf)
        P_F = logits[..., : self.output_dim]  # * (1 - x) + x * -100

        # Mask forward
        mask_forward = torch.zeros(self.seq_len, self.num_pressures, 1)  # .to(self.device)
        mask_forward[t, pr] = 1
        mask_forward[0] = 0

        P_F = (
            P_F.reshape(-1, self.seq_len, self.num_pressures, self.num_tokens) * (mask_forward)
            + (1 - mask_forward) * -1000
        )

        mask_pr = self.vocab.mask_alpha[pr]

        P_F[:, t, pr] = P_F[:, t, pr] * mask_pr + (1 - mask_pr) * -1000

        return P_F.view(-1)


class FlowGenerator(nn.Module):
    def __init__(
        self,
        pressure_set: Mapping[str, List],
        seq_len: int,
        gamma: float = 0.5,
        delta: float = 0.1,
        lr: float = 3e-4,
        minibatch_size: int = 32,
        num_hid: int = 512,
        test: bool = False,
        verbose: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self.pressure_set = pressure_set
        self.seq_len = seq_len

        self.num_pressures = len(pressure_set)
        self.vocab = Vocab(alphabet=pressure_set)
        self.num_tokens = self.vocab.num_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = TBModelClosedLoop(
            self.vocab, self.seq_len, self.num_pressures, self.num_tokens, num_hid
        )
        self.opt = torch.optim.Adam(self.model.parameters(), lr)

        self.gamma = gamma
        self.delta = delta
        self.minibatch_size = minibatch_size

        self.test = test
        self.verbose = verbose
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.experience_buffer = None

    def initialize_state(self, init_protocol: Mapping[str, float]) -> torch.Tensor:

        tokenized_protocol = tokenize_protocol([init_protocol], self.vocab)
        protocol_tensor = torch.tensor(tokenized_protocol, dtype=torch.long)
        init_pressures = F.one_hot(protocol_tensor, num_classes=self.num_tokens)
        init_state = torch.zeros(self.seq_len, self.num_pressures, self.num_tokens)
        init_state[0] = init_pressures

        return init_state

    def one_hot_encode(self, protocols: torch.Tensor) -> torch.Tensor:
        protocols_one_hot = torch.zeros(protocols.shape[0], self.seq_len, self.num_pressures, self.num_tokens)

        for i in range(protocols_one_hot.shape[0]):
            token_tensor = tokenize_tensor(protocols[i], self.vocab)
            protocols_one_hot[i] = F.one_hot(token_tensor, num_classes=self.num_tokens)

        return protocols_one_hot

    def get_initial_state(self) -> torch.Tensor:
        # Each episode starts with an "empty state" except for the first
        state = torch.zeros(self.seq_len, self.num_pressures, self.num_tokens)
        state[0] = self.experience_buffer[0][0]

        return state

    def check_state(self, state: torch.Tensor) -> bool:
        num_tokens = state.shape[-1]
        ones = torch.ones(state.view(-1, num_tokens).shape[0])

        if (ones == torch.sum(state.view(-1, num_tokens), dim=1)).all().item():
            return True
        else:
            return False

    def train(
        self,
        protocols: torch.Tensor,
        reward_func: Callable,
        num_episodes: int = 50000,
        t_start: int = 1,
        early_stop_tolerance: int = 100,
        tb_losses: list = [],
        logZs: list = [],
    ) -> None:

        # One hot encode dataset and build experience buffer
        self.experience_buffer = self.one_hot_encode(protocols)

        tb_mean_loss = []
        tb_early_stop_loss = []
        best_loss = 1.0e6
        early_stop_count = 0

        self.model.train()

        for episode in tqdm.tqdm(range(num_episodes), ncols=40):

            # -------------
            # Training Loop
            # -------------
            minibatch_trajs = self.generate_minibatch()
            dataloader = DataLoader(
                minibatch_trajs,
                batch_size=self.minibatch_size,
                shuffle=False,
            )

            for i, minibatch_trajs in enumerate(dataloader):

                # Each episode starts with an "empty state" except for the first
                state = torch.zeros(
                    len(minibatch_trajs), self.seq_len, self.num_pressures, self.num_tokens
                )  # .to(self.device)
                state[:, 0] = minibatch_trajs[:, 0]

                # Define total P_F
                # total_P_F = 0
                total_P_F = torch.zeros(len(minibatch_trajs))  # .to(self.device)
                rewards = torch.zeros(len(minibatch_trajs))  # .to(self.device)

                for t in range(t_start, self.seq_len):
                    for pr in range(self.num_pressures):

                        P_F_s = self.model(state, t, pr)
                        action = torch.argmax(minibatch_trajs[:, t, pr], dim=1)

                        # "Go" to the next state
                        new_state = state.detach().clone()
                        new_state[:, t, pr] = F.one_hot(action, num_classes=self.num_tokens)

                        P_F_s_reshape = P_F_s.reshape(-1, self.seq_len, self.num_pressures, self.num_tokens)
                        cat = Categorical(logits=P_F_s_reshape[:, t, pr])

                        # Accumulate the P_F sum
                        total_P_F += cat.log_prob(action)

                        if t == self.seq_len - 1 and pr == self.num_pressures - 1:
                            # # If we've built a complete state.
                            # check = self.check_state(new_state)
                            # if check:
                            for r in range(len(rewards)):
                                rewards[r] = reward_func(
                                    torch.from_numpy(state_to_protocol(minibatch_trajs[r], self.vocab)).float()
                                )
                            # else:
                            #     raise ValueError("State is not valid")

                        # Continue to next iteration with updated state
                        state = new_state

                # We're done with the trajectory, let's compute its loss. Since the reward can
                # sometimes be zero, instead of log(0) we'll clip the log-reward to -20.
                # Using modified equation (11) from Ref. Jain et. al  since total_P_B = 0

                loss = (self.model.logZ + total_P_F - torch.log(rewards).clip(-20)).pow(2).mean()

                tb_losses.append(loss.item())
                tb_early_stop_loss.append(loss.item())
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
                logZs.append(self.model.logZ.item())

            if episode % 100 == 0:
                train_loss = np.mean(tb_early_stop_loss)
                tb_mean_loss.append(train_loss)
                tb_early_stop_loss = []

                if train_loss < best_loss:
                    best_loss = train_loss
                    best_params = [i.data.numpy() for i in self.model.parameters()]
                    early_stop_count = 0
                    if self.verbose:
                        print(f"Train Loss decrease: {best_loss}")
                else:
                    early_stop_count += 1

                if early_stop_count >= early_stop_tolerance:
                    print(best_loss)
                    print("Early stopping...")
                    break

        for p, best_p in zip(self.model.parameters(), best_params):
            p.data = torch.tensor(best_p)

        return tb_losses, tb_mean_loss, logZs

    # def run_heuristic_checks(self, state, trajectories):
    #     # Check if protocol already exists in trajectories
    #     if len(trajectories) > 0:
    #         protocol_exists = utils.check_protocol_buffer(state, trajectories)
    #     else:
    #         protocol_exists = False

    #     if protocol_exists:
    #         return protocol_exists

    #     # Check if protocol already ran in experience buffer
    #     protocol_ran = utils.check_protocol_buffer(state, self.experience_buffer)

    #     if protocol_ran:
    #         return protocol_ran

    #     # Check for consecutive repeating envs if not in test
    #     if not self.test:
    #         repeating = utils.check_repeating_environments(state)

    #         if repeating:
    #             return repeating

    def generate_minibatch(self, t_start: int = 1):

        # Define torch tensor for online trajectories
        trajectories = torch.tensor([])

        # Define m_online: select gamma*m online data
        m_online = int(self.gamma * self.minibatch_size)
        max_attempts = m_online * 10
        attempts = 0

        self.model.eval()

        while len(trajectories) < m_online:
            attempts += 1
            state = self.get_initial_state()  # .to(self.device)
            state = state.unsqueeze(0)

            if attempts >= max_attempts:
                if self.verbose:
                    print("Max attempts reached.")

                break

            for t in range(t_start, self.seq_len):
                for pr in range(self.num_pressures):
                    # mask_pr = self.vocab.mask_pr[pr].to(self.device)

                    P_F_s = self.model(state, t, pr)

                    # Here P_F is logits, so we want the Categorical to compute the softmax for us
                    cat = Categorical(logits=P_F_s)

                    if torch.rand(1) < self.delta:
                        action = torch.tensor(self.rng.choice(torch.where(P_F_s != -1000)[0]), dtype=int)
                    else:
                        action = cat.sample()

                    # "Go" to the next state
                    # state.view(-1, self.seq_len * self.num_pressures * self.num_tokens)
                    new_state = state.detach().clone().view(-1, self.seq_len * self.num_pressures * self.num_tokens)
                    new_state[:, action] = 1
                    new_state = new_state.reshape(-1, self.seq_len, self.num_pressures, self.num_tokens)

                    # Continue to next iteration with updated state
                    state = new_state

            # state = state.squeeze(0)

            # # Check if protocol already exists in trajectories or experience buffer
            # heuristic_check = self.run_heuristic_checks(state, trajectories)

            # if heuristic_check:
            #     continue

            # # Check for consecutive repeating envs if not in test mode
            # if not self.test:
            #     repeating = utils.check_repeating_environments(state.squeeze(0))

            #     if repeating:
            #         continue

            # Check if we've built a complete state.
            check = self.check_state(state)
            if check:
                # Append the trajectory to the list
                trajectories = torch.cat((trajectories, state), dim=0)
            else:
                raise ValueError("State is not valid")

        # Generate a random permutation of indices and choose the first 'num_choices' elements
        m_offline = self.minibatch_size - len(trajectories)
        offline_indices = torch.randperm(len(self.experience_buffer))[:m_offline]
        trajs_offline = self.experience_buffer[offline_indices]

        # Concatenate the online and offline trajectories
        minibatch_trajs = torch.cat((trajs_offline, trajectories), dim=0)

        return minibatch_trajs

    def sample(self, sample_len: int = 1000, t_start: int = 1):
        with torch.no_grad():
            self.model.eval()

            tb_sampled_protocols = []  # torch.zeros(sample_len, self.seq_len, self.num_pressures)

            max_attempts = sample_len * 5
            attempts = 0

            # for s in range(sample_len):
            while len(tb_sampled_protocols) < sample_len:
                attempts += 1
                state = self.get_initial_state()  # .to(self.device)

                if attempts >= max_attempts:
                    if self.verbose:
                        print("Max attempts reached.")

                    break

                for t in range(t_start, self.seq_len):
                    for pr in range(self.num_pressures):
                        # mask_pr = self.vocab.mask_pr[pr].to(self.device)

                        P_F_s = self.model(state, t, pr)

                        # Here P_F is logits, so we want the Categorical to compute the softmax for us
                        cat = Categorical(logits=P_F_s)
                        action = cat.sample()

                        # "Go" to the next state
                        new_state = state.detach().clone().view(-1)
                        new_state[action] = 1
                        new_state = new_state.reshape(self.seq_len, self.num_pressures, self.num_tokens)

                        # Continue to next iteration with updated state
                        state = new_state

                # if not self.test:
                #     # Check if protocol already exists in trajectories or experience buffer
                #     heuristic_check = self.run_heuristic_checks(state, tb_sampled_protocols)

                #     if heuristic_check:
                #         continue

                # Append sampled protocol to list
                tb_sampled_protocols.append(state)

            if len(tb_sampled_protocols) > 0:
                sampled_protocols = np.array(
                    [state_to_protocol(state, self.vocab) for state in tb_sampled_protocols],
                )
            else:
                sampled_protocols = np.array([])

            return sampled_protocols
