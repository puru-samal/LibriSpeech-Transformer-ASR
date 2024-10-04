import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import *

class Encoder(torch.nn.Module):
    def __init__(self, input_dim, num_layers, d_model, num_heads, d_ff, mfcc_max_seq_length, enc_dropout=0.1):
        super(Encoder, self).__init__()

        self.embedding      = ResBlockEmbedding(cin=input_dim, cout=d_model, stride=2, down_sample=True)
        self.pos_encoding   = PositionalEncoding(d_model, mfcc_max_seq_length)
        self.enc_layers     = torch.nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, enc_dropout) for _ in range(num_layers)])
        self.dropout        = torch.nn.Dropout(enc_dropout)

    def forward(self, x):
        encoder_output = self.embedding(x)
        input_lengths = torch.tensor([len(i) for i in encoder_output])
        
        # apply Positional Encoding on these extracted features
        x = self.pos_encoding(encoder_output)

        # apply dropout as regularization technique
        x = self.dropout(x)

        # passing inputs through Transformer Encoder blocks
        for layer in self.enc_layers:
            x, self_attention_weights = layer(x)

        return x, input_lengths, self_attention_weights
    

class ASREncoder(nn.Module):
    def __init__(self, input_dim, num_layers, d_model, num_heads, d_ff, mfcc_max_seq_length, num_classes, enc_dropout=0.1, blank_idx=0):
        super(ASREncoder, self).__init__()

        self.encoder = Encoder(input_dim, num_layers, d_model, num_heads, d_ff, mfcc_max_seq_length, enc_dropout)
        # Linear layer to project from d_model to the number of output classes (characters or phonemes)
        self.output_layer = nn.Linear(d_model, num_classes)
        # Log softmax layer to get log probabilities
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.blank_idx = blank_idx  # Index for the CTC blank token


    def forward(self, x):
        encoder_output, input_lengths, self_attention_weights = self.encoder(x)
        # Apply linear projection to map to number of classes
        output = self.output_layer(encoder_output)
        # Apply log softmax for CTC loss
        output = self.log_softmax(output)
        return output, input_lengths, self_attention_weights


    def inference(self, x, input_lengths, idx_to_char, mode="greedy", timesteps=50, p=0.95, device='cuda'):
        """
        Perform inference on input x and return decoded sequences.
        Supports greedy decoding and nucleus sampling.
        idx_to_char: A dictionary mapping from class indices to characters/phonemes.
        mode: Choose between "greedy" and "nucleus".
        p: Nucleus sampling threshold (only used for nucleus sampling).
        timesteps: Number of timesteps to predict for nucleus sampling.
        """
        if mode == "greedy":
            # Perform greedy decoding
            outputs, _, _ = self.forward(x, input_lengths)
            decoded_indices = self.greedy_decode(outputs)
            decoded_transcriptions = [''.join([idx_to_char[idx] for idx in seq]) for seq in decoded_indices]
            return decoded_transcriptions

        elif mode == "nucleus":
            # Perform nucleus sampling
            log_prob, decoded_sequences = self.predict_nucleus_sampling(x, timesteps, idx_to_char, p)
            return decoded_sequences


    def greedy_decode(self, outputs):
        """
        Greedy decoding of the output of the network (logits).
        outputs: (batch_size, max_seq_length, num_classes)

        Returns a list of predicted sequences (indices of classes).
        """
        # Get the argmax at each timestep (most probable class)
        predicted_indices = torch.argmax(outputs, dim=-1)  # Shape: (batch_size, max_seq_length)

        decoded_sequences = []
        for batch_idx in range(predicted_indices.size(0)):  # For each sequence in the batch
            pred_seq = predicted_indices[batch_idx].tolist()

            # Remove consecutive duplicates and blanks
            decoded_seq = []
            previous_idx = self.blank_idx  # Initialize with blank to handle leading blanks

            for idx in pred_seq:
                if idx != previous_idx and idx != self.blank_idx:  # Skip blanks and consecutive repeats
                    decoded_seq.append(idx)
                previous_idx = idx

            decoded_sequences.append(decoded_seq)

        return decoded_sequences
    

    def predict_nucleus_sampling(self, x, timesteps, p=0.95):
        import torch.nn.functional as F
        import torch

        x = torch.tensor(x).long()
        batch_size, seq_len = x.shape

        # Initialize sequences and log probabilities
        seq = x
        log_prob = torch.zeros(batch_size)
        prob_dists = []

        for _ in range(timesteps):  # We only need to predict the remaining timesteps
            with torch.inference_mode():
                y, _ = self.forward(seq)
                last_prob = y[:, -1, :]  # Get last time step output
                prob_dists.append(last_prob)  # Store distribution
                # Apply softmax to convert logits to probabilities
                probs = F.softmax(last_prob, dim=-1)
                # Sort the probabilities and their corresponding indices
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                # Calculate cumulative probabilities
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                # Select indices where cumulative probability <= p
                top_p_mask = cumulative_probs <= p
                top_p_mask[:, 0] = True  # Ensure at least one token is selected
                # Mask out probabilities outside the nucleus (i.e., top-p subset)
                top_p_probs = sorted_probs * top_p_mask
                # Normalize the probabilities within the top-p set
                normalized_top_p_probs = top_p_probs / top_p_probs.sum(dim=-1, keepdim=True)
                # Sample the next token from the top-p subset
                sampled_indices = torch.multinomial(normalized_top_p_probs, 1).squeeze(1)
                # Map sampled indices back to the original token space
                next_token = sorted_indices.gather(1, sampled_indices.unsqueeze(1)).squeeze(1)
                # Update sequence with the newly sampled token
                seq = torch.cat((seq, next_token.unsqueeze(1)), dim=1)
                # Update log probabilities
                log_prob = log_prob + probs.gather(1, next_token.unsqueeze(1)).log().squeeze(1)

        # Extract only the predicted part, removing the original input
        predicted_seq = seq[:, seq_len:]
        return log_prob, predicted_seq

    
    