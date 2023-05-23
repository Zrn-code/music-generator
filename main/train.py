# PYTORCH
import glob
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
from music21 import converter, instrument, note, chord
from get_note import get_notes
from create_model import CombinedModel,MusicDataset,MusicModel,create_network
import os

def train(model,epoch, dataloader_notes, dataloader_offsets, dataloader_durations, n_vocab_notes, n_vocab_offsets, n_vocab_durations):
    """ train the neural network """
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters())
    model.to(device)
    
    for epoch in range(epoch):
        running_loss = 0.0
        model.train()

        for (notes, _), (offsets, _), (durations, _) in zip(dataloader_notes, dataloader_offsets, dataloader_durations):
            notes = notes.to(device)
            offsets = offsets.to(device)
            durations = durations.to(device)
            notes_output, offsets_output, durations_output = model.forward(notes, offsets, durations)
            
            notes_loss = criterion(notes_output.view(-1, n_vocab_notes), notes.view(-1).long())
            offsets_loss = criterion(offsets_output.view(-1, n_vocab_offsets), offsets.view(-1).long())
            durations_loss = criterion(durations_output.view(-1, n_vocab_durations), durations.view(-1).long())

            loss = notes_loss + offsets_loss + durations_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch} Loss: {running_loss / len(dataloader_notes)}")
        model_path = f'model-{epoch}.pt'
        torch.save(model.state_dict(), model_path)
        
def train_network(epoch,preprocess):
    """ Train a Neural Network to generate music """
    notes, offsets, durations = get_notes(preprocess)

    n_vocab_notes = len(set(notes))
    n_vocab_offsets = len(set(offsets))
    n_vocab_durations = len(set(durations))

    dataset_notes = MusicDataset(notes, n_vocab_notes)
    dataset_offsets = MusicDataset(offsets, n_vocab_offsets)
    dataset_durations = MusicDataset(durations, n_vocab_durations)

    dataloader_notes = DataLoader(dataset_notes, batch_size=1024, shuffle=True)
    dataloader_offsets = DataLoader(dataset_offsets, batch_size=1024, shuffle=True)
    dataloader_durations = DataLoader(dataset_durations, batch_size=1024, shuffle=True)

    model = create_network(n_vocab_notes, n_vocab_offsets, n_vocab_durations)
    print("Start training")
    train(model,epoch, dataloader_notes, dataloader_offsets, dataloader_durations, n_vocab_notes, n_vocab_offsets, n_vocab_durations)

if __name__ == '__main__':
    #config = get_argument()
    #epoch,preprocess = config['epoch'],config['get_note']
    train_network(50,1)
