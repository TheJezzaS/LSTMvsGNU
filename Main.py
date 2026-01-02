import os
import math
from datetime import datetime
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

'''
Params for lstm given in the paper were, quote
Our best performing non-regularized
LSTM has two hidden layers with 200 units per layer, and its weights are initialized uniformly in
[−0.1, 0.1]. We train it for 4 epochs with a learning rate of 1 and then we decrease the learning rate
by a factor of 2 after each epoch, for a total of 13 training epochs. The size of each minibatch is 20,
and we unroll the network for 20 steps.
'''

## Use cuda if releven, otherwise mps (if mac), otherwise cpu (sad)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


############################################
# PTB DATA LOADING
############################################
class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus:
    def __init__(self, path):
        self.dictionary = Dictionary()
        # run tokenize for all datasets, not just train, to make sure when we test and validate there arent errors of
        # 'ive never seen this word, my dict doesnt know how to handle it'
        # note only the training will be used ot train, but all should be in the dict
        self.train = self.tokenize(os.path.join(path, "ptb.train.txt"))
        self.valid = self.tokenize(os.path.join(path, "ptb.valid.txt"))
        self.test  = self.tokenize(os.path.join(path, "ptb.test.txt"))

    def tokenize(self, path):
        assert os.path.exists(path)

        # first pass over the file; create the dict
        with open(path, "r") as f:
            tokens = 0
            for line in f:
                words = line.split() + ["<eos>"] # add end of sequence for each line
                tokens += len(words)
                for w in words:
                    self.dictionary.add_word(w)

        ids = torch.LongTensor(tokens) # 1D tensor of word indices, ie one very long sentence
        idx = 0
        # second pass, use the dict to convert the text to a tensor
        with open(path, "r") as f:
            for line in f:
                words = line.split() + ["<eos>"]
                for w in words:
                    ids[idx] = self.dictionary.word2idx[w]
                    idx += 1
        return ids # tensor of words converted to int's


def batchify(data, batch_size):
    """
    Convert a 1D tensor of token indices into a 2D tensor suitable for RNN input.

    The output tensor has:
        - Each **column** = a separate sequence (batch)
        - Each **row** = a time step

    Args:
        data (torch.Tensor): 1D tensor of token indices, shape (num_tokens,)
        batch_size (int): number of sequences (columns) to create

    Returns:
        torch.Tensor: 2D tensor of shape (num_batches, batch_size),
                      where rows = time steps, columns = batch sequences

    Example:
         data = torch.arange(1, 13)  # 1D tensor: [1,2,...,12]
         batch_size = 3
         device = torch.device("cpu")
         batch = batchify(data, batch_size, device)
         print(batch)
        tensor([[ 1,  5,  9],
                [ 2,  6, 10],
                [ 3,  7, 11],
                [ 4,  8, 12]])

    Explanation of example:
        - Input tensor: [1,2,3,4,5,6,7,8,9,10,11,12]
        - Batch size = 3 → split into 3 sequences
        - Each row = time step, each column = batch sequence
          Column 0 → [1,2,3,4]
          Column 1 → [5,6,7,8]
          Column 2 → [9,10,11,12]
    """

    nbatch = data.size(0) // batch_size # compute how many full batches we can make
    data = data.narrow(0, 0, nbatch * batch_size)    # trim data to fit exactly into full batches
    data = data.view(batch_size, -1)    # reshape into 2D tensor with batch_size rows
    data = data.t()    # transpose so each row = time step
    data = data.contiguous()    # make memory contiguous
    data = data.to(device)    # move to device
    return data



def get_batch(source, i, bptt): # bptt = backprop through time length
    # slides window of size bptt over source matrix
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i : i+seq_len]
    target = source[i+1 : i+1+seq_len].reshape(-1)
    return data, target

############################################
# MODEL
############################################
class RNNModel(nn.Module):
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout):
        # ntoken = num unique words
        # ninp = num dimentions in the 'word space' (200 for us, following the paper)
        # nhid = num dimentions of hidden state, h_t
        # nlayers = num layers (2 according to paper)
        super().__init__()
        self.drop = nn.Dropout(dropout)

        # Converts word IDs into learnable N-dimensional vectors (embeddings) that the LSTM uses as input
        # takes in the IDs vector (words converted to int's) and maps each to a randomly initialised 200d vector
        self.encoder = nn.Embedding(ntoken, ninp)

        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(ninp, nhid, nlayers, dropout=dropout)
        else:
            raise ValueError("Invalid rnn type")

        # fc layer to convert hidden state into prob of each possible word as next word
        self.decoder = nn.Linear(nhid, ntoken)
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

        self.init_weights()

    def init_weights(self):
        initrange = 0.1 # according to paper
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        """
        This is the model-level forward, which includes:
            1. Converting word IDs to embeddings
            2. Passing embeddings through LSTM/GRU layers
            3. Applying dropout
            4. Decoding hidden states to vocabulary logits

        Args:
            input: LongTensor of shape (seq_len, batch_size)
                   Sequence of word IDs from get_batch
            hidden: tuple (h, c) of LSTM hidden and cell states (recall h is long term mem, c is short)
                   Each of shape (num_layers, batch_size, nhid)
                   (for GRU, this would just be h)

        Returns:
            decoded: FloatTensor of shape (seq_len*batch_size, ntoken)
                     Logits for each word in the vocabulary at each time step
            hidden: updated tuple (h_n, c_n) (or just h_n for GRU)
                    Hidden states to feed into the next batch/sequence
        """

        # Convert word IDs to embeddings and apply dropout
        # input -> (seq_len, batch_size)
        # emb -> (seq_len, batch_size, ninp)
        # Each word ID is mapped to a vector
        emb = self.drop(self.encoder(input))

        # Pass embeddings through the LSTM/gru
        # output -> (seq_len, batch_size, nhid)
        # hidden -> (h_n, c_n), each (num_layers, batch_size, nhid)
        # LSTM/gru processes the sequence and keeps track of context
        output, hidden = self.rnn(emb, hidden)

        # Apply dropout to LSTM outputs to reduce overfitting
        # Shape stays the same: (seq_len, batch_size, nhid)
        output = self.drop(output)

        # Flatten output to feed into decoder
        # output.view(...) -> (seq_len*batch_size, nhid)
        # Each hidden vector is now a separate example for the linear layer
        # decoder -> (seq_len*batch_size, ntoken)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))

        # Return logits and updated hidden states
        return decoded, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == "LSTM":
            return (weight.new_zeros(self.nlayers, batch_size, self.nhid),
                    weight.new_zeros(self.nlayers, batch_size, self.nhid))
        else: # GRU
            return weight.new_zeros(self.nlayers, batch_size, self.nhid)


############################################
# TRAIN and EVAL
############################################
def train_epoch(model, data, optimizer, criterion, bptt):
    model.train() # set to train mode to allow dropout
    total_loss = 0
    hidden = model.init_hidden(data.size(1)) # init hidden state
                                             # For LSTM: (h, c)
                                             # For GRU:  (h)

    for batch_start_idx in range(0, data.size(0) - 1, bptt): # loop over data in sizes of bptt
        inputs, targets = get_batch(data, batch_start_idx, bptt)
        optimizer.zero_grad() # Clear previous grads before current batch
        output, hidden = model(inputs, hidden)
        loss = criterion(output, targets)
        loss.backward()


        # the paper gives clipping for larger models:
        # (units, clipping) = (650, 5), (1500, 10)
        # so by linear interpolation, (200, 2.35)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.35)
        optimizer.step() #Update model parameters using gradients

        total_loss += loss.item()
        hidden = tuple(h.detach() for h in hidden) if isinstance(hidden, tuple) else hidden.detach()

    return math.exp(total_loss / (len(data) / bptt))


def evaluate(model, data, criterion, bptt):
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(data.size(1))

    with torch.no_grad():
        for batch_start_idx in range(0, data.size(0) - 1, bptt):
            inputs, targets = get_batch(data, batch_start_idx, bptt)
            output, hidden = model(inputs, hidden)
            loss = criterion(output, targets)
            total_loss += loss.item()
            hidden = tuple(h.detach() for h in hidden) if isinstance(hidden, tuple) else hidden.detach()

    return math.exp(total_loss / (len(data) / bptt))


############################################
# EXPERIMENT RUNNER
############################################
def run_experiment(rnn_type, dropout, label):
    base_lr = 1.0
    epochs = 13

    model = RNNModel(
        rnn_type=rnn_type,
        ntoken=ntokens,
        ninp=200,
        nhid=200,
        nlayers=2,
        dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr)

    train_ppl, test_ppl, val_ppl= [], [], []

    for epoch in range(1, epochs + 1):

        # Zaremba LR decay according to paper
        if epoch > 4:
            lr = base_lr * (0.5 ** (epoch - 4))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        else:
            lr = base_lr

        tr = train_epoch(model, train_data, optimizer, criterion, bptt)
        te = evaluate(model, test_data, criterion, bptt)
        val = evaluate(model, valid_data, criterion, bptt)

        train_ppl.append(tr)
        test_ppl.append(te)
        val_ppl.append(val)

        print(
            f"{label} | Epoch {epoch:2d} | "
            f"LR {lr:.4f} | Train PPL {tr:.2f} | Valid PPL {val:.2f} | Test PPL {te:.2f}"
        )

    return train_ppl, val_ppl, test_ppl, base_lr, dropout


############################################
# MAIN
############################################
if __name__ == "__main__":
    torch.manual_seed(1)

    corpus = Corpus("PTB")
    ntokens = len(corpus.dictionary)

    batch_size = 20
    bptt = 20 # according to paper

    train_data = batchify(corpus.train, batch_size)
    test_data  = batchify(corpus.test, batch_size)
    valid_data = batchify(corpus.valid, batch_size)

    experiments = [
        ("LSTM", 0.0, "LSTM no dropout"),
        ("LSTM", 0.5, "LSTM dropout"),
        ("GRU",  0.0, "GRU no dropout"),
        ("GRU",  0.5, "GRU dropout"),
    ]

    results = {}

    exp_start_time = datetime.now()
    for rnn_type, dropout, label in experiments:
        results[label] = run_experiment(rnn_type, dropout, label)

    print(f'All Experiments finished, took {format(datetime.now() - exp_start_time)}')
    print("\nFinal Perplexities (Train / Valid / Test):")
    for label, (train_ppl, val_ppl, test_ppl, lr, dropout) in results.items():
        print(f"{label:15s} | {train_ppl[-1]:6.2f} / {val_ppl[-1]:6.2f} / {test_ppl[-1]:6.2f}")

    ########################################
    # PLOTTING (4 plots, 8 curves)
    ########################################

    linestyles = ["-", "--", ":"] # "-" for Train, "--" for Test

    plt.figure(figsize=(12, 7))

    for idx, (label, (train_ppl, val_ppl, test_ppl, lr, dropout)) in enumerate(results.items()):
        plt.plot(train_ppl, label=f"{label} Train (LR={lr}, keep_prob={1 - dropout})", linestyle=linestyles[0])
        plt.plot(test_ppl, label=f"{label} Test (LR={lr}, keep_prob={1 - dropout})", linestyle=linestyles[1])
        plt.plot(val_ppl, label=f"{label} Valid (LR={lr}, keep_prob={1 - dropout})", linestyle=linestyles[2])

    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("PTB Experiments: LSTM/GRU, small 200-unit network")
    plt.legend()
    plt.grid(True)

    plt.savefig("all_experiments.png", dpi=300)
    plt.show()


