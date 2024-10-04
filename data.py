import torch
from torch.nn.utils.rnn import pad_sequence
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import torchaudio.transforms as tat

VOCAB = [
   "<sos>", "<eos>",
    "A",   "B",    "C",    "D",
    "E",   "F",    "G",    "H",
    "I",   "J",    "K",    "L",
    "M",   "N",    "O",    "P",
    "Q",   "R",    "S",    "T",
    "U",   "V",    "W",    "X",
    "Y",   "Z",    "'",    " ", "<pad>"
]

def print_data_stats(dataset, dataloader, batch_size, header="Train Dataset"):
    print(header)
    print("#MFCCs       : ", dataset.__len__())
    print("Max MFCC len : ", dataset.max_mfcc_len)
    print("Batch Size   : ", batch_size)
    print("#Batches     : ", dataloader.__len__())
    print("Shapes of the Data --")
    for batch in dataloader:
        if dataset.partition == 'test-clean':
            x_pad, x_len = batch
            print(f"x_pad shape:\t\t{x_pad.shape}")
            print(f"x_len shape:\t\t{x_len.shape}")
            break

        x_pad, y_shifted_pad, y_golden_pad, x_len, y_len, = batch
        print(f"x_pad shape:\t\t{x_pad.shape}")
        print(f"x_len shape:\t\t{x_len.shape}")
        print(f"y_shifted_pad shape:\t{y_shifted_pad.shape}")
        print(f"y_golden_pad shape:\t{y_golden_pad.shape}")
        print(f"y_len shape:\t\t{y_len.shape}")
        break
    print("\n")

    return x_pad


def plot_random_mfccs(dataloader, num_samples=5, aspect='auto'):
    """
    Randomly samples `num_samples` MFCCs from a batch in a DataLoader and plots them.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader to sample from.
        num_samples (int): Number of MFCCs to sample and plot (default is 5).
    """
    inputs = next(iter(dataloader))
    inputs = inputs[0]
    batch_size = inputs.shape[0]
    if num_samples > batch_size:
        raise ValueError(f"The batch size ({batch_size}) is smaller than the requested number of samples ({num_samples}).")
    
    sample_indices = random.sample(range(batch_size), num_samples)

    fig, axs = plt.subplots(3, 3, figsize=(10, 10))

    # Plot the sampled MFCCs
    idx = 0
    for i in range(3):
        for j in range(3):
            if idx < num_samples:
                if inputs[idx].ndim == 1:
                    mfcc = inputs[sample_indices[idx]].reshape(1, -1)
                else:
                    mfcc = inputs[sample_indices[idx]]
                axs[i, j].imshow(np.flipud(mfcc.T), cmap='viridis', aspect=aspect)  # Using the specified colormap and aspect ratio
                axs[i, j].set_title(f"MFCC {sample_indices[idx]}")
                axs[i, j].axis('off')
                idx += 1
            else:
                axs[i, j].axis('off')  # Hide the last plot if we have fewer samples than grid slots

    plt.tight_layout()
    plt.show()



class SpeechDataset(torch.utils.data.Dataset):
    ''' 
    Train | Val | Test dataloaders
    '''
    def __init__(self, root_dir, partition, subset=1.0, vocab=VOCAB, print_vocab=True, augument=True, 
                 time_mask_width=30, time_mask_p=0.3, freq_mask_width=10, cepstral=True):

        self.partition       = partition
        self.augument        = augument
        self.time_mask_width = time_mask_width
        self.time_mask_p     = time_mask_p
        self.freq_mask_width = freq_mask_width
        self.time_masking    = tat.TimeMasking(time_mask_param=self.time_mask_width, iid_masks=True, p=self.time_mask_p)
        self.freq_masking    = tat.FrequencyMasking(freq_mask_param=self.freq_mask_width, iid_masks=True)
        
        # mfcc
        self.mfcc_dir           = os.path.join(root_dir, partition, "mfcc")
        self.mfcc_files         = sorted(os.listdir(self.mfcc_dir))
        self.mfccs              = []
        
        # transcripts
        if partition != 'test-clean':
            self.transcript_dir     = os.path.join(root_dir, partition, "transcripts")
            self.transcript_files   = sorted(os.listdir(self.transcript_dir))
            self.transcripts_shifted, self.transcripts_golden  = [], []
            assert len(self.mfcc_files) == len(self.transcript_files)

        # vocab
        self.length             = int(len(self.mfcc_files) * subset)
        self.vocab              = vocab
        self.VOCAB_MAP          = {self.vocab[i]:i for i in range(0, len(VOCAB))}
        self.PAD_TOKEN          = self.VOCAB_MAP["<pad>"]
        self.SOS_TOKEN          = self.VOCAB_MAP["<sos>"]
        self.EOS_TOKEN          = self.VOCAB_MAP["<eos>"]

        if print_vocab:
            print(f"Length of Vocabulary    : {len(self.vocab)}")
            print(f"PAD_TOKEN               : {self.PAD_TOKEN}")
            print(f"SOS_TOKEN               : {self.SOS_TOKEN}")
            print(f"EOS_TOKEN               : {self.EOS_TOKEN}")
            print("\n")

        print("Loaded Path: ", partition)
        self.max_mfcc_len = 0
        for i in range(self.length):
            
            # mfcc
            mpath = os.path.join(self.mfcc_dir, self.mfcc_files[i])
            mfcc = np.load(mpath)
            if cepstral:
                mfcc = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1E-8)
            self.mfccs.append(mfcc)
            self.max_mfcc_len = max(self.max_mfcc_len, mfcc.shape[0])

            if partition != 'test-clean':
                # transcript
                tpath = os.path.join(self.transcript_dir, self.transcript_files[i])
                temp = np.load(tpath)[1:-1]
                self.transcripts_shifted.append(np.array([self.SOS_TOKEN] + [self.vocab.index(i) for i in temp]))
                self.transcripts_golden.append(np.array([self.vocab.index(i) for i in temp] + [self.EOS_TOKEN]))
            
        if partition != 'test-clean':
            assert len(self.mfccs) == len(self.transcripts_shifted)


    def __len__(self): return self.length


    def __getitem__(self, ind):
        mfcc                = torch.FloatTensor(self.mfccs[ind])
        if self.partition == 'test-clean':
            return mfcc
        shifted_transcript  = torch.tensor(self.transcripts_shifted[ind])
        golden_transcript   = torch.tensor(self.transcripts_golden[ind])
        return mfcc, shifted_transcript, golden_transcript
       


    def collate_fn(self, batch):
        '''
        1.  Extract the features and labels from 'batch'.
        2.  We will additionally need to pad both features and labels,
        3.  Perform transforms, if you so wish.
        4.  Train|Val : Return batch of features, labels, lengths of features, and lengths of labels
            Test      : Return batch of features, lengths of features
        '''

        # Batch of input mfcc coefficients.
        batch_mfcc              = [i[0] for i in batch]
        batch_mfcc_pad          = pad_sequence(batch_mfcc, batch_first=True)
        lengths_mfcc            = [len(i) for i in batch_mfcc]
    
        # Batch of output characters (shifted and golden).
        if self.partition != 'test-clean':
            batch_transcript        = [i[1] for i in batch]
            batch_golden            = [i[2] for i in batch]
            lengths_transcript      = [len(i) for i in batch_transcript]
            batch_transcript_pad    = pad_sequence(batch_transcript, batch_first=True, padding_value=self.PAD_TOKEN)
            batch_golden_pad        = pad_sequence(batch_golden, batch_first=True, padding_value=self.PAD_TOKEN)

        
        if self.augument:
            batch_mfcc_pad = (self.freq_masking(self.time_masking(batch_mfcc_pad.T))).T
        
        # Return the following values:
        # padded features, padded shifted labels, padded golden labels, actual length of features, actual length of the shifted labels
        if self.partition == 'test-clean':
            return batch_mfcc_pad, torch.tensor(lengths_mfcc)
        return batch_mfcc_pad, batch_transcript_pad, batch_golden_pad, torch.tensor(lengths_mfcc), torch.tensor(lengths_transcript)