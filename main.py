import torch
from torchsummaryX import summary
from data import SpeechDataset, print_data_stats, plot_random_mfccs
from utils import *

import gc
import yaml
import os
import math

import random
import zipfile
import datetime
import wandb
import numpy as np
import pandas as pd

from encoder import *
import warnings
warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)
print("\n")


###### Config -----------------------------------------------------------------
EXPT_CONFIG_FILE = 'config_base.yaml'
with open(EXPT_CONFIG_FILE) as file:
    config = yaml.safe_load(file)


###### Datasets -----------------------------------------------------------------
train_dataset   = SpeechDataset(
    root_dir    = config['data_root'],
    partition   = config['train_partition'],
    subset      = 1.0,
    print_vocab = True,
    augument    = True,
    cepstral    = config['cepstral_norm'],
    time_mask_width = config['time_mask_width'],
    time_mask_p     = config['time_mask_p'],
    freq_mask_width = config['freq_mask_width'], 
)

val_dataset     = SpeechDataset(
    root_dir    = config['data_root'],
    partition   = config['val_partition'],
    subset      = 1.0,
    print_vocab = False,
    augument    = False,
    cepstral    = config['cepstral_norm']
)

test_dataset    = SpeechDataset(
    root_dir    = config['data_root'],
    partition   = config['test_partition'],
    subset      = 1.0,
    print_vocab = False,
    augument    = False,
    cepstral    = config['cepstral_norm']
)

###### Dataloaders -----------------------------------------------------------------
train_loader    = torch.utils.data.DataLoader(
    dataset     = train_dataset,
    batch_size  = config["batch_size"],
    shuffle     = True,
    num_workers = 4,
    pin_memory  = True,
    collate_fn  = train_dataset.collate_fn
)

val_loader      = torch.utils.data.DataLoader(
    dataset     = val_dataset,
    batch_size  = config["batch_size"],
    shuffle     = False,
    num_workers = 4,
    pin_memory  = True,
    collate_fn  = val_dataset.collate_fn,
)

test_loader     = torch.utils.data.DataLoader(
    dataset     = test_dataset,
    batch_size  = config["batch_size"],
    shuffle     = False,
    num_workers = 2,
    pin_memory  = True,
    collate_fn  = test_dataset.collate_fn,
)

print("\n")
dummy_input = print_data_stats(train_dataset, train_loader, config["batch_size"], header="Train:")
print_data_stats(val_dataset,   val_loader,   config["batch_size"], header="Val:")
print_data_stats(test_dataset,  test_loader,  config["batch_size"], header="Test:")

#plot_random_mfccs(train_loader, num_samples=9)
#plot_random_mfccs(train_loader, num_samples=9)
#plot_random_mfccs(test_loader, num_samples=9)
gc.collect()

###### Encoder -----------------------------------------------------------------
model = ASREncoder(input_dim = config["input_dim"],
                    num_layers= config["enc_num_layers"],
                    num_heads = config["enc_num_heads"],
                    d_model  = config["d_model"],
                    d_ff  = config["d_ff"],
                    mfcc_max_seq_length = 3000,
                    num_classes  = config['num_classes']).to(device)

para = num_parameters(model)
print("#"*10)
print(f"Model Parameters:\n {para}")
print("#"*10)


#summary = summary(model, dummy_input.to(device))

###### Loss -----------------------------------------------------------------
loss_func   = nn.CTCLoss()
scaler      = torch.cuda.amp.GradScaler()


###### Optim -----------------------------------------------------------------
if config["optimizer"] == "SGD":
  optimizer = torch.optim.SGD(model.parameters(),
                              lr=config["learning_rate"],
                              momentum=config["momentum"],
                              weight_decay=1E-4,
                              nesterov=config["nesterov"])
elif config["optimizer"] == "Adam":
  optimizer = torch.optim.Adam(model.parameters(),
                               lr=float(config["learning_rate"]),
                               weight_decay=1e-4)
elif config["optimizer"] == "AdamW":
  optimizer = torch.optim.AdamW(model.parameters(),
                                lr=float(config["learning_rate"]),
                                weight_decay=0.01)


###### Scheduler -----------------------------------------------------------------
if config["scheduler"] == "ReduceLR":
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                factor=config["factor"], patience=config["patience"], min_lr=1E-8, verbose=True)
elif config["scheduler"] == "CosineAnnealing":
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                T_max = config["epochs"], eta_min=1E-8)
  


###### Training -----------------------------------------------------------------
e                   = 0
best_loss           = 20

checkpoint_root = os.path.join(os.getcwd(), "encoder")
os.makedirs(checkpoint_root, exist_ok=True)
# wandb.watch(model, log="all")

checkpoint_best_loss_model_filename     = 'checkpoint-best-loss-model.pth'
checkpoint_last_epoch_filename          = 'checkpoint-epoch-'
best_loss_model_path                    = os.path.join(checkpoint_root, checkpoint_best_loss_model_filename)

# if RESUME_LOGGING:
#     # change if you want to load best test model accordingly
#     checkpoint = torch.load(wandb.restore(checkpoint_best_loss_model_filename, run_path=""+run_id).name)

#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     e = checkpoint['epoch']

#     print("Resuming from epoch {}".format(e+1))
#     print("Epochs left: ", config['epochs']-e)
#     print("Optimizer: \n", optimizer)

torch.cuda.empty_cache()
gc.collect()

idx_to_char = {v:k for k,v in train_dataset.VOCAB_MAP.items()}
VOCAB       = train_dataset.vocab

epochs = config["epochs"]
for epoch in range(e, epochs):

    print("\nEpoch {}/{}".format(epoch+1, config["epochs"] ))

    curr_lr = float(optimizer.param_groups[0]["lr"])

    train_loss, train_perplexity, attention_weights = train_model(model, train_loader, optimizer, loss_func, scaler, device)

    print("\nEpoch {}/{}: \nTrain Loss {:.04f}\t Train Perplexity {:.04f}\t Learning Rate {:.04f}".format(
        epoch + 1, config["epochs"], train_loss, train_perplexity, curr_lr))

    if (epoch % 5 == 0):    # validate every 2 epochs to speed up training
        # After training, you can validate the model using the following calls:
        attn = attention_weights.cpu().detach().numpy()
        plot_attention(attn)
        visualize_attention(attention_weights[None, :, ])

        # Greedy Decoding Validation
        running_distance = validate_and_save_predictions(model, val_loader, idx_to_char=idx_to_char, vocab=VOCAB, mode="greedy", output_file="greedy_predictions.txt")

        # Nucleus Sampling Validation
#       running_distance = validate_and_save_predictions(model, val_loader, VOCAB, vocab=VOCAB, mode="nucleus", p=0.95, timesteps=50, output_file="nucleus_predictions.txt")

        print("Levenshtein Distance {:.04f}".format(running_distance))

#         wandb.log({"train_loss"     : train_loss,
#                 "train_perplexity"  : train_perplexity,
#                 "learning_rate"     : curr_lr,
#                 "val_distance"      : levenshtein_distance})

    else:
#         wandb.log({"train_loss"     : train_loss,
#                 "train_perplexity"  : train_perplexity,
#                 "learning_rate"     : curr_lr})

        pass

    if config["scheduler"] == "ReduceLR":
        scheduler.step(running_distance)
    else:
        scheduler.step()

    ### Highly Recommended: Save checkpoint in drive and/or wandb if accuracy is better than your current best
    epoch_model_path = os.path.join(checkpoint_root, (checkpoint_last_epoch_filename + str(epoch) + '.pth'))
    save_model(model, optimizer, scheduler, ['train_loss', train_loss], epoch, epoch_model_path)
    ## wandb.save(epoch_model_path) ## Can't save on wandb for all epochs, may blow up storage

    print("Saved epoch model")

    if train_loss <= best_loss:
        best_loss = train_loss
        save_model(model, optimizer, scheduler, ['train_loss', train_loss], epoch, best_loss_model_path)
        # wandb.save(best_loss_model_path)
        print("Saved best training model")

## Finish your wandb run
#run.finish()

