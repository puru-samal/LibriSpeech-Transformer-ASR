import os
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def num_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1E6

# Function to save the predictions to a text file
def save_predictions_to_file(predictions, targets, file_path, mode="greedy"):
    with open(file_path, "a") as f:
        for idx, (target, pred) in enumerate(zip(targets, predictions)):
            f.write(f"Target | Output #{idx} ({mode}): {target} | {pred}\n")

def validate_and_save_predictions(model, dataloader, idx_to_char, vocab, mode="greedy", p=0.95, timesteps=50, output_file="predictions.txt", device='cuda'):
    model.eval()

    # Progress bar
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc="Val", ncols=5)

    running_distance = 0.0
    with open(output_file, 'w'):  # Clear the file before appending predictions
        pass

    for i, (inputs, targets_shifted, targets_golden, inputs_lengths, targets_lengths) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets_golden = targets_golden.to(device)

        with torch.inference_mode():
            # Perform greedy or nucleus sampling based on the mode selected
            if mode == "greedy":
                predictions = model.inference(inputs, inputs_lengths, idx_to_char, mode="greedy")
            elif mode == "nucleus":
                print(inputs.shape)
                predictions = model.inference(inputs, inputs_lengths, idx_to_char, mode="nucleus", timesteps=timesteps, p=p, device=device)

        # Convert target indices to characters
        target_strings = ["".join(indices_to_chars(targets_golden[batch_idx, 0:targets_lengths[batch_idx]], vocab)) for batch_idx in range(len(inputs))]

        # Save predictions and target to file
        save_predictions_to_file(predictions, target_strings, output_file, mode)

        # Calculate Levenshtein Distance

        running_distance += calc_edit_distance(predictions, inputs.shape[0], targets_golden, targets_lengths, vocab, print_example=False)

        # Online validation distance monitoring
        batch_bar.set_postfix(
            running_distance="{:.04f}".format(float(running_distance / (i + 1)))
        )
        batch_bar.update()

        del inputs, targets_shifted, targets_golden, inputs_lengths, targets_lengths
        torch.cuda.empty_cache()

    batch_bar.close()
    running_distance /= len(dataloader)

    return running_distance

def indices_to_chars(indices, vocab, SOS_TOKEN=0, EOS_TOKEN=1, PAD_TOKEN=30):
    tokens = []
    for i in indices:  # looping through all indices
        if int(i) == SOS_TOKEN:     # If SOS is encountered, don't add it to the final list
            continue
        elif int(i) == EOS_TOKEN or int(i) == PAD_TOKEN:   # If EOS is encountered, stop the decoding process
            break
        else:
            tokens.append(vocab[i])
    return tokens

# Function to calculate Levenshtein distance
def calc_edit_distance(predictions,batch_size, y, y_len, vocab, print_example=False):
    dist = 0.0
#     batch_size, seq_len = predictions.shape

    for batch_idx in range(batch_size):
        y_sliced = predictions[batch_idx]
        pred_sliced = predictions[batch_idx]

        # strings - when you are using characters from the SpeechDataset
        y_string = "".join(y_sliced)
        pred_string = "".join(pred_sliced)

        # Uncomment to use Levenshtein distance calculation, if you have a function/library
        # dist += Levenshtein.distance(pred_string, y_string)

        if print_example:
            print("\nGround Truth : ", y_string)
            print("Prediction   : ", pred_string)

    dist /= batch_size
    return dist


def calculate_ctc_loss(loss_func, out, targets, input_lengths, target_lengths):
    """
    Calculate CTC loss for ASR task.

    Parameters:
    - loss_func: torch.nn.CTCLoss instance.
    - out: Output logits from the model (before softmax), expected shape (B, T, Vocab_size).
    - targets: Target sequences (concatenated together), shape (total_target_length).
    - input_lengths: Lengths of each input in the batch.
    - target_lengths: Lengths of each target sequence.

    Returns:
    - CTC loss.
    """

    # CTC Loss expects (T, B, Vocab_size), so we need to transpose out
    out = out.transpose(0, 1)  # Convert from (B, T, Vocab_size) to (T, B, Vocab_size)

    # Calculate CTC Loss
    loss = loss_func(out, targets, input_lengths, target_lengths)

    return loss

def plot_attention(attention):
    # Function for plotting attention
    # You need to get a diagonal plot
    plt.clf()
    sns.heatmap(attention, cmap='GnBu')
    plt.show()

def calculate_entropy(att_weights):
    epsilon = 1e-10 
    att_weights = torch.clamp(att_weights, epsilon, 1 - epsilon)  
    entropy = -torch.sum(att_weights * torch.log(att_weights), dim=-1)
    return entropy.mean() 




def visualize_attention(attention_weights, index=0):
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights[index].detach().cpu().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title(f'Attention Weights for Sample {index}')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.show()


def train_model(model, train_loader, optimizer, loss_func, scaler, device):
    model.train()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc="Train")

    total_loss = 0
    running_loss = 0.0
    running_perplexity = 0.0

    for i, (inputs, _, targets, inputs_lengths, targets_lengths) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
        inputs_lengths = inputs_lengths.to(device)
        targets_lengths = targets_lengths.to(device)

        with torch.cuda.amp.autocast():
            logits, inputs_lengths, attention_weights = model(inputs)
            loss = calculate_ctc_loss(loss_func, logits, targets, inputs_lengths, targets_lengths)

        # Backward pass and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        # Track loss and perplexity
        running_loss += float(loss.item())
        perplexity = torch.exp(loss)
        running_perplexity += perplexity.item()

        # Update progress bar
        batch_bar.set_postfix(
            loss="{:.04f}".format(float(running_loss / (i + 1))),
            perplexity="{:.04f}".format(float(running_perplexity / (i + 1)))
        )
        batch_bar.update()

        # Clean up to free memory
        del inputs, targets, inputs_lengths, targets_lengths
        torch.cuda.empty_cache()

    # Calculate average loss and perplexity over the epoch
    running_loss = float(running_loss / len(train_loader))
    running_perplexity = float(running_perplexity / len(train_loader))

    batch_bar.close()
    return running_loss, running_perplexity, attention_weights


def save_model(model, optimizer, scheduler, metric, epoch, path):
    torch.save(
        {"model_state_dict"         : model.state_dict(),
         "optimizer_state_dict"     : optimizer.state_dict(),
         "scheduler_state_dict"     : scheduler.state_dict() if scheduler is not None else {},
         metric[0]                  : metric[1],
         "epoch"                    : epoch},
         path
    )

def load_model(path, model, metric= "valid_acc", optimizer= None, scheduler= None):

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer != None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler != None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch   = checkpoint["epoch"]
    metric  = checkpoint[metric]

    return [model, optimizer, scheduler, epoch, metric]