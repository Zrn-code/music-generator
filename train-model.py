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

def get_argument():
    # do not modify
    opt = argparse.ArgumentParser()
    opt.add_argument("-g","--get_note",
                     type=int,
                     default=1,
                     help="whether preprocessing, 0 means no and 1 means yes")
    opt.add_argument("-e","--epoch",
                     type=int,
                     default=50,
                     help="The number of epoch")

    config = vars(opt.parse_args()) 
    return config

class MusicDataset(Dataset):
    def __init__(self, notes, n_vocab):
        self.sequence_length = 100
        self.pitchnames = sorted(set(item for item in notes))
        self.note_to_int = dict((note, number) for number, note in enumerate(self.pitchnames))
        self.network_input, self.network_output = self.prepare_sequences(notes, n_vocab)

    def __len__(self):
        return len(self.network_input)

    def __getitem__(self, idx):
        return self.network_input[idx], self.network_output[idx]

    def prepare_sequences(self, notes, n_vocab):
        network_input = []
        network_output = []

        for i in range(len(notes) - self.sequence_length):
            sequence_in = notes[i:i + self.sequence_length]
            sequence_out = notes[i + self.sequence_length]
            network_input.append([self.note_to_int[char] for char in sequence_in])
            network_output.append(self.note_to_int[sequence_out])

        n_patterns = len(network_input)

        network_input = torch.tensor(network_input).reshape(n_patterns, self.sequence_length, 1).float()
        network_input = network_input / float(n_vocab)
        
        # One-hot encode the network_output
        network_output = torch.tensor(network_output).long()
        network_output = network_output.unsqueeze(1)
        network_output = nn.functional.one_hot(network_output, num_classes=n_vocab).float()

        return network_input, network_output
    
def get_notes(preprocess):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []
    offsets = []
    durations = []
    if preprocess == 1:
        for file in glob.glob("classical-piano-type0/*.mid"):
            midi = converter.parse(file)

            print("Parsing %s" % file)

            notes_to_parse = None

            try: # file has instrument parts
                s2 = instrument.partitionByInstrument(midi)
                notes_to_parse = s2.parts[0].recurse() 
            except: # file has notes in a flat structure
                notes_to_parse = midi.flat.notes

            offsetBase = 0
            for element in notes_to_parse:
                isNoteOrChord = False
                
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                    isNoteOrChord = True
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
                    isNoteOrChord = True

                if isNoteOrChord:
                    offsets.append(str(element.offset - offsetBase))
                    durations.append(str(element.duration.quarterLength))
                    isNoteOrChord = False
                    offsetBase = element.offset

        with open('data/notes', 'wb') as filepath:
            pickle.dump(notes, filepath)

        with open('data/durations', 'wb') as filepath:
            pickle.dump(durations, filepath)

        with open('data/offsets', 'wb') as filepath:
            pickle.dump(offsets, filepath)

        return notes, offsets, durations
    else:
        with open('data/notes', 'rb') as filepath:
            notes = pickle.load(filepath)

        with open('data/durations', 'rb') as filepath:
            durations = pickle.load(filepath)

        with open('data/offsets', 'rb') as filepath:
            offsets = pickle.load(filepath)

        return notes, offsets, durations
class CombinedModel(nn.Module):
    def __init__(self, model_notes, model_offsets, model_durations):
        super(CombinedModel, self).__init__()
        self.model_notes = model_notes
        self.model_offsets = model_offsets
        self.model_durations = model_durations

    def forward(self, note_input, offset_input, duration_input):
        notes_out = self.model_notes(note_input)
        offsets_out = self.model_offsets(offset_input)
        durations_out = self.model_durations(duration_input)
        return notes_out, offsets_out, durations_out

class MusicModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MusicModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        out = self.linear(out)  
        return out
    
def create_network(n_vocab_notes, n_vocab_offsets, n_vocab_durations):
    input_size = 1
    hidden_size = 256
    output_size_notes = n_vocab_notes
    output_size_offsets = n_vocab_offsets
    output_size_durations = n_vocab_durations

    model_notes = MusicModel(input_size, hidden_size, output_size_notes)
    model_offsets = MusicModel(input_size, hidden_size, output_size_offsets)
    model_durations = MusicModel(input_size, hidden_size, output_size_durations)

    model = CombinedModel(model_notes, model_offsets, model_durations)
    #model.load_state_dict(torch.load('model-8.pt'))
    return model

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

    dataloader_notes = DataLoader(dataset_notes, batch_size=64, shuffle=True)
    dataloader_offsets = DataLoader(dataset_offsets, batch_size=64, shuffle=True)
    dataloader_durations = DataLoader(dataset_durations, batch_size=64, shuffle=True)

    model = create_network(n_vocab_notes, n_vocab_offsets, n_vocab_durations)
    train(model,epoch, dataloader_notes, dataloader_offsets, dataloader_durations, n_vocab_notes, n_vocab_offsets, n_vocab_durations)

if __name__ == '__main__':
    config = get_argument()
    epoch,preprocess = config['epoch'],config['get_note']
    train_network(epoch,preprocess)
