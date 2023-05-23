from torch.utils.data import Dataset
import torch.nn as nn
import torch

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
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        out = self.linear(out) 
        return out

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
    #model.load_state_dict(torch.load(f'model-{epoch-1}.pt'))
    return model