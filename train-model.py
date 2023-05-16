# PYTORCH
import glob
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from music21 import converter, instrument, note, chord

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
        network_output = np.eye(n_vocab)[network_output]

        return network_input, network_output

class MusicModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MusicModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        out = self.fc(out)
        return out
    
    
def get_notes():
	""" Get all the notes and chords from the midi files in the ./midi_songs directory """
	notes = []
	offsets = []
	durations = []
	
	for file in glob.glob("classical-piano-type0/*.mid"):
		midi = converter.parse(file)

		#print("Parsing %s" % file)

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



def train(model, dataloader_notes, dataloader_offsets, dataloader_durations):
    """ train the neural network """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    model.to(device)

    for epoch in range(10):
        running_loss = 0.0

        for i, ((notes, _), (offsets, _), (durations, _)) in enumerate(zip(dataloader_notes, dataloader_offsets, dataloader_durations)):
            notes = notes.to(device).float()
            offsets = offsets.to(device).float()
            durations = durations.to(device).float()

            optimizer.zero_grad()

            notes_output = model[0](notes)
            offsets_output = model[1](offsets)
            durations_output = model[2](durations)
            notes_loss = criterion(notes_output.view(-1, notes_output.shape[2]), torch.flatten(torch.argmax(notes, dim=2)))
            offsets_loss = criterion(offsets_output.view(-1, offsets_output.shape[2]), torch.flatten(torch.argmax(offsets, dim=2)))
            durations_loss = criterion(durations_output.view(-1, durations_output.shape[2]), torch.flatten(torch.argmax(durations, dim=2)))

            loss = notes_loss + offsets_loss + durations_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch {epoch+1} Loss: {running_loss / len(dataloader_notes)}")
        model_path = f'model-{epoch}-{running_loss / len(dataloader_notes):.4f}.pt'
        torch.save(model.state_dict(), model_path)
        
    

def train_network():
    """ Train a Neural Network to generate music """
    notes, offsets, durations = get_notes()

    n_vocab_notes = len(set(notes))
    n_vocab_offsets = len(set(offsets))
    n_vocab_durations = len(set(durations))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_notes = MusicDataset(notes, n_vocab_notes)
    dataset_offsets = MusicDataset(offsets, n_vocab_offsets)
    dataset_durations = MusicDataset(durations, n_vocab_durations)

    dataloader_notes = DataLoader(dataset_notes, batch_size=64, shuffle=True)
    dataloader_offsets = DataLoader(dataset_offsets, batch_size=64, shuffle=True)
    dataloader_durations = DataLoader(dataset_durations, batch_size=64, shuffle=True)

    model = create_network(n_vocab_notes, n_vocab_offsets, n_vocab_durations)
    model.to(device)

    train(model, dataloader_notes, dataloader_offsets, dataloader_durations)
    
def create_network(n_vocab_notes, n_vocab_offsets, n_vocab_durations):
    input_size = 1
    hidden_size = 256
    output_size_notes = n_vocab_notes
    output_size_offsets = n_vocab_offsets
    output_size_durations = n_vocab_durations
    model = nn.Sequential(
    MusicModel(input_size, hidden_size, output_size_notes),
    MusicModel(input_size, hidden_size, output_size_offsets),
    MusicModel(input_size, hidden_size, output_size_durations)
    )
    return model

if __name__ == '__main__':
	train_network()