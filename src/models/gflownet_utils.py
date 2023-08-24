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
    """Standard Dataset class that inherits attributes from Dataset.

    Parameters
    ----------
    x: torch.Tensor
        Dataset of sequences.
    y: torch.Tensor
        Scores.
    """
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        super().__init__()
        self.x = x
        self.y = y

        self.n_samples = self.x.shape[0]

    def __getitem__(self, index: int) -> Tuple:
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


# Making tokenizer
class Vocab(object):
    """Vocab class that contains objects for tokenization.

    Parameters
    ----------
    alphabet: Mapping[str, List]
        Dictionary containing alphabet
    """

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
        """Function for building tokenization dictionaries.        
        """

        # Build vocabulary and find max alphabet length
        alphabet_to_token = {}
        token_to_alphabet = {}
        alphabet_index = {}
        index_alphabet = {}

        for index, key in enumerate(self.alphabet):
            alphabet_to_i = {}
            i_to_alphabet = {}

            for i, alpha in enumerate(self.alphabet[key]):
                alphabet_to_i[alpha] = i
                i_to_alphabet[i] = alpha

            alphabet_to_token[key] = alphabet_to_i
            token_to_alphabet[key] = i_to_alphabet
            alphabet_index[key] = index
            index_alphabet[index] = key

            self.max_alphabet_len = max(self.max_alphabet_len, len(self.alphabet[key]))

        return alphabet_to_token, token_to_alphabet, alphabet_index, index_alphabet

    def build_mask(self, alphabet_index):
        """
        Function for building a mask dictionary.
        """
        # Build mask
        mask_alpha = {}
        
        for key in self.alphabet:
            mask_alphabet = torch.zeros(self.max_alphabet_len)
            mask_alphabet[: len(self.alphabet[key])] = 1
            mask_alpha[alphabet_index[key]] = mask_alphabet

        return mask_alpha


def state_to_sequence(one_hot: torch.Tensor, vocab: Vocab) -> np.ndarray:
    """
    Function that takes one-hot encoded state and returns sequence.
    """

    sequence = []
    state_max_arg = torch.argmax(one_hot, dim=2).numpy()

    for alphabet_t in state_max_arg:
        alphabet = []
        for i, token in enumerate(alphabet_t):
            alphabet.append(vocab.token_to_alphabet[vocab.index_alphabet[i]][token])
        sequence.append(alphabet)

    return np.array(sequence)


def tokenize_sequence(sequence: List[Dict[str, float]], vocab: Vocab) -> List:
    """
    Function that accepts a list of keys and values and returns a list of tokenized_sequences.
    """

    tokenized_sequence = []

    for alphabet_t in sequence:
        tokenized_alphabet = []
        for key in alphabet_t.keys():
            tokenized_alphabet.append(vocab.alphabet_to_token[key][alphabet_t[key]])
        tokenized_sequence.append(tokenized_alphabet)

    return tokenized_sequence


def tokenize_tensor(tensor: torch.Tensor, vocab: Vocab) -> torch.Tensor:
    """
    Function that accepts a sequence tensor and returns a tokenized tensor.
    """

    tokenized_tensor = torch.zeros(tensor.shape[0], tensor.shape[1], dtype=int)

    for i, alphabet_t in enumerate(tensor):
        for j, alpha in enumerate(alphabet_t):
            projected_alpha_list = np.array(vocab.alphabet[vocab.index_alphabet[j]])
            distances = np.abs(projected_alpha_list - alpha.item())
            min_arg = np.argmin(distances)
            projection = projected_alpha_list[min_arg]
            tokenized_tensor[i, j] = vocab.alphabet_to_token[vocab.index_alphabet[j]][projection]

    return tokenized_tensor


def to_numpy(sequence_dict: List[Dict[str, int]], alphabet_index) -> np.ndarray:
    """
    Function that accepts sequence dictionary and returns a numpy sequence.
    """

    sequence_np = np.full((len(sequence_dict), len(sequence_dict[0])), np.nan)

    for t in range(len(sequence_dict)):
        alphabets = sequence_dict[t]
        for key in alphabets.keys():
            sequence_np[t, alphabet_index[key]] = alphabets[key]

    return sequence_np


def get_possible_sequences(alphabets: Mapping[str, List], seq_len: int, init_sequence: OrderedDict):
    """
    Function that uses a param_grid to generate all possible sequences of a given length for a given alphabet.
    """

    param_grid = {key: alphabets[key] for key in alphabets}
    possible_sets = list(ParameterGrid(param_grid))

    full_grid = list(product(possible_sets, repeat=seq_len))
    possible_sequences = [sequence for sequence in full_grid if sequence[0] == init_sequence]

    return possible_sequences


def compute_score_distribution(
    alphabets: Mapping[str, List],
    score_func: Callable,
    seq_len: int,
    init_sequence: OrderedDict,
):
    """
    Function that accepts alphabets, score_func, seq_len, and initial sequence and returns
    the expected and observed score distributions.
    """

    # Get possible sequences
    possible_sequences = get_possible_sequences(alphabets, seq_len, init_sequence)

    # Compute expected reward distribution
    score_distribution = {}
    empirical_score_distribution = {}
    tensor_to_key = {}
    dataset = []

    for i, sequence in enumerate(possible_sequences):
        sequence_to_numpy = np.array([np.array(list(alpha.values())) for alpha in sequence])
        sequence_tensor = torch.from_numpy(sequence_to_numpy).float()
        dataset.append(sequence_to_numpy)
        tensor_to_key[str(sequence_to_numpy)] = i
        score_distribution[i] = score_func(sequence_tensor)
        empirical_score_distribution[i] = 0

    return score_distribution, empirical_score_distribution, np.array(dataset), tensor_to_key


class TBModelClosedLoop(nn.Module):
    """
    Parameters
    ----------
    vocab: Vocab
        Instantiated vocabulary class containing tokenization and masking objects.
    seq_len: int
        Sequence length.
    num_alphabets: int
        Number of alphabets represented in the state.
    num_tokens: int
        Number of tokens (alphabet with max number of tokens).
    num_hid: int
        Number of hidden layers.
    """

    def __init__(self, vocab: Vocab, seq_len: int, num_alphabets: int, num_tokens: int, num_hid: int):
        super().__init__()
        # The input dimension is seq_len * num_alphabets * num_tokens.
        self.vocab = vocab
        self.seq_len = seq_len
        self.num_alphabets = num_alphabets
        self.num_tokens = num_tokens

        self.input_dim = self.seq_len * self.num_alphabets * self.num_tokens
        self.output_dim = self.seq_len * self.num_alphabets * self.num_tokens

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, num_hid),
            nn.LeakyReLU(),
            nn.Linear(num_hid, self.output_dim),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # logZ is just a single number
        self.logZ_nn = nn.Parameter(torch.ones(1))

    @property
    def logZ(self):
        return self.logZ_nn

    def forward(self, state, t, alpha):

        input_mlp = state.view(-1, self.seq_len * self.num_alphabets * self.num_tokens)
        logits = self.mlp(input_mlp)

        # Slice the logits, and mask invalid actions (since we're predicting
        # log-values), we use -100 since exp(-100) is tiny, but we don't want -inf)
        P_F = logits[..., : self.output_dim]  # * (1 - x) + x * -100

        # Mask forward
        mask_forward = torch.zeros(self.seq_len, self.num_alphabets, 1)  # .to(self.device)
        mask_forward[t, alpha] = 1
        mask_forward[0] = 0

        P_F = (
            P_F.reshape(-1, self.seq_len, self.num_alphabets, self.num_tokens) * (mask_forward)
            + (1 - mask_forward) * -1000
        )

        mask_alpha = self.vocab.mask_alpha[alpha]

        P_F[:, t, alpha] = P_F[:, t, alpha] * mask_alpha + (1 - mask_alpha) * -1000

        return P_F.view(-1)


class FlowGenerator(nn.Module):
    """
    Implementation of GFlowNet training procedure based on Ref:
    Jain, Moksh, et al. "Biological sequence design with gflownets." International Conference on Machine Learning. PMLR, 2022.

    Parameters
    ----------
    alphabet: Mapping[str, List]
        Dictionary containing alphabet
    seq_len: int
        Sequence length.
    gamma: float
        Proportion of online sequences to use for training.
    delta: float
        Exploration parameter that controls degree of sampling random actions.
    lr: float
        Learning rate for training.
    minibatch_size: int
        Minibatch size.
    num_hid: int
        Number of hidden layers to use in MLP.
    verbose: bool
        Flag for running training in verbose.
    seed: int
        Seed for setting up rng.
    """

    def __init__(
        self,
        alphabet: Mapping[str, List],
        seq_len: int,
        gamma: float = 0.5,
        delta: float = 0.1,
        lr: float = 3e-4,
        minibatch_size: int = 32,
        num_hid: int = 512,
        verbose: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self.alphabet = alphabet
        self.seq_len = seq_len

        self.num_alphabet = len(alphabet)
        self.vocab = Vocab(alphabet=alphabet)
        self.num_tokens = self.vocab.num_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = TBModelClosedLoop(
            self.vocab, self.seq_len, self.num_alphabet, self.num_tokens, num_hid
        )
        self.opt = torch.optim.Adam(self.model.parameters(), lr)

        self.gamma = gamma
        self.delta = delta
        self.minibatch_size = minibatch_size

        self.verbose = verbose
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.offline_data = None

    def initialize_state(self, init_sequence: Mapping[str, float]) -> torch.Tensor:

        tokenized_sequence = tokenize_sequence([init_sequence], self.vocab)
        sequence_tensor = torch.tensor(tokenized_sequence, dtype=torch.long)
        init_alphabet = F.one_hot(sequence_tensor, num_classes=self.num_tokens)
        init_state = torch.zeros(self.seq_len, self.num_alphabet, self.num_tokens)
        init_state[0] = init_alphabet

        return init_state

    def one_hot_encode(self, sequences: torch.Tensor) -> torch.Tensor:
        sequences_one_hot = torch.zeros(sequences.shape[0], self.seq_len, self.num_alphabet, self.num_tokens)

        for i in range(sequences_one_hot.shape[0]):
            token_tensor = tokenize_tensor(sequences[i], self.vocab)
            sequences_one_hot[i] = F.one_hot(token_tensor, num_classes=self.num_tokens)

        return sequences_one_hot

    def get_initial_state(self) -> torch.Tensor:
        # Each episode starts with an "empty state" except for the first
        state = torch.zeros(self.seq_len, self.num_alphabet, self.num_tokens)
        state[0] = self.offline_data[0][0]

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
        sequences: torch.Tensor,
        score_func: Callable,
        num_episodes: int = 50000,
        t_start: int = 1,
        early_stop_tolerance: int = 100,
        tb_losses: list = [],
        logZs: list = [],
    ) -> None:

        # One hot encode dataset and build experience buffer
        self.offline_data = self.one_hot_encode(sequences)

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
                    len(minibatch_trajs), self.seq_len, self.num_alphabet, self.num_tokens
                )
                state[:, 0] = minibatch_trajs[:, 0]

                # Define total P_F
                # total_P_F = 0
                total_P_F = torch.zeros(len(minibatch_trajs))
                scores = torch.zeros(len(minibatch_trajs))

                for t in range(t_start, self.seq_len):
                    for alpha in range(self.num_alphabet):

                        P_F_s = self.model(state, t, alpha)
                        action = torch.argmax(minibatch_trajs[:, t, alpha], dim=1)

                        # "Go" to the next state
                        new_state = state.detach().clone()
                        new_state[:, t, alpha] = F.one_hot(action, num_classes=self.num_tokens)

                        P_F_s_reshape = P_F_s.reshape(-1, self.seq_len, self.num_alphabet, self.num_tokens)
                        cat = Categorical(logits=P_F_s_reshape[:, t, alpha])

                        # Accumulate the P_F sum
                        total_P_F += cat.log_prob(action)

                        if t == self.seq_len - 1 and alpha == self.num_alphabet - 1:
                            for r in range(len(scores)):
                                scores[r] = score_func(
                                    torch.from_numpy(state_to_sequence(minibatch_trajs[r], self.vocab)).float()
                                )

                        # Continue to next iteration with updated state
                        state = new_state

                # We're done with the trajectory, let's compute its loss. Since the reward can
                # sometimes be zero, instead of log(0) we'll clip the log-reward to -20.
                # Using modified equation (11) from Ref. Jain et. al  since total_P_B = 0

                loss = (self.model.logZ + total_P_F - torch.log(scores).clip(-20)).pow(2).mean()

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
                for alpha in range(self.num_alphabet):
                    P_F_s = self.model(state, t, alpha)

                    # Here P_F is logits, so we want the Categorical to compute the softmax for us
                    cat = Categorical(logits=P_F_s)

                    if torch.rand(1) < self.delta:
                        action = torch.tensor(self.rng.choice(torch.where(P_F_s != -1000)[0]), dtype=int)
                    else:
                        action = cat.sample()

                    # "Go" to the next state
                    # state.view(-1, self.seq_len * self.num_alphabet * self.num_tokens)
                    new_state = state.detach().clone().view(-1, self.seq_len * self.num_alphabet * self.num_tokens)
                    new_state[:, action] = 1
                    new_state = new_state.reshape(-1, self.seq_len, self.num_alphabet, self.num_tokens)

                    # Continue to next iteration with updated state
                    state = new_state

            # Check if we've built a complete state.
            check = self.check_state(state)
            if check:
                # Append the trajectory to the list
                trajectories = torch.cat((trajectories, state), dim=0)
            else:
                raise ValueError("State is not valid")

        # Generate a random permutation of indices and choose the first 'num_choices' elements
        m_offline = self.minibatch_size - len(trajectories)
        offline_indices = torch.randperm(len(self.offline_data))[:m_offline]
        trajs_offline = self.offline_data[offline_indices]

        # Concatenate the online and offline trajectories
        minibatch_trajs = torch.cat((trajs_offline, trajectories), dim=0)

        return minibatch_trajs

    def sample(self, sample_len: int = 1000, t_start: int = 1):
        with torch.no_grad():
            self.model.eval()

            tb_sampled_sequences = []

            max_attempts = sample_len * 5
            attempts = 0

            while len(tb_sampled_sequences) < sample_len:
                attempts += 1
                state = self.get_initial_state()

                if attempts >= max_attempts:
                    if self.verbose:
                        print("Max attempts reached.")

                    break

                for t in range(t_start, self.seq_len):
                    for alpha in range(self.num_alphabet):

                        P_F_s = self.model(state, t, alpha)

                        # Here P_F is logits, so we want the Categorical to compute the softmax for us
                        cat = Categorical(logits=P_F_s)
                        action = cat.sample()

                        # "Go" to the next state
                        new_state = state.detach().clone().view(-1)
                        new_state[action] = 1
                        new_state = new_state.reshape(self.seq_len, self.num_alphabet, self.num_tokens)

                        # Continue to next iteration with updated state
                        state = new_state

                # Append sampled sequence to list
                tb_sampled_sequences.append(state)

            if len(tb_sampled_sequences) > 0:
                sampled_sequences = np.array(
                    [state_to_sequence(state, self.vocab) for state in tb_sampled_sequences],
                )
            else:
                sampled_sequences = np.array([])

            return sampled_sequences
