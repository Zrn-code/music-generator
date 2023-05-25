import torch
import torch.nn.functional as F
from music21 import note, chord, instrument, stream
from get_note import get_notes
from create_model import create_network, MusicDataset
import numpy as np
import fractions
import argparse

def generate_notes(model, network_input_notes, network_input_offsets, network_input_durations, notenames, offsetnames, durationames, n_vocab_notes, n_vocab_offsets, n_vocab_durations,length):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    start = np.random.randint(0, len(network_input_notes) - 1)
    start2 = np.random.randint(0, len(network_input_offsets) - 1)
    start3 = np.random.randint(0, len(network_input_durations) - 1)

    int_to_note = dict((number, note) for number, note in enumerate(notenames))
    print(int_to_note)
    int_to_offset = dict((number, note) for number, note in enumerate(offsetnames))
    int_to_duration = dict((number, note) for number, note in enumerate(durationames))

    pattern = torch.tensor(network_input_notes[start])
    pattern2 = torch.tensor(network_input_offsets[start2])
    pattern3 = torch.tensor(network_input_durations[start3])
    prediction_output = []

    # generate notes or chords
    for note_index in range(length):
        note_prediction_input = pattern.unsqueeze(0)
        note_prediction_input = note_prediction_input / float(n_vocab_notes)

        offset_prediction_input = pattern2.unsqueeze(0)
        offset_prediction_input = offset_prediction_input / float(n_vocab_offsets)

        duration_prediction_input = pattern3.unsqueeze(0)
        duration_prediction_input = duration_prediction_input / float(n_vocab_durations)

        prediction = model(note_prediction_input, offset_prediction_input, duration_prediction_input)
        prediction = [F.softmax(pred, dim=1) for pred in prediction]
        unknown_count = 0
        try:
            index = torch.multinomial(prediction[0].reshape(-1), num_samples=1).item()
            result = int_to_note[index]
        except KeyError:
            unknown_count += 1
            continue
        
        try:
            offset = torch.multinomial(prediction[1].reshape(-1), num_samples=1).item()
            offset_result = int_to_offset[offset]
        except KeyError:
            unknown_count += 1
            continue
        
        try:
            duration = torch.multinomial(prediction[2].reshape(-1), num_samples=1).item()
            duration_result = int_to_duration[duration]
        except KeyError:
            unknown_count += 1
            continue
        
        print("Next note: " + str(result) + " - Duration: " + str(duration_result) + " - Offset: " + str(offset_result))

        prediction_output.append([result, offset_result, duration_result])

        pattern = torch.cat((pattern, torch.tensor([index]).reshape((1, 1))))
        pattern2 = torch.cat((pattern2, torch.tensor([offset]).reshape((1, 1))))
        pattern3 = torch.cat((pattern3, torch.tensor([duration]).reshape((1, 1))))
        pattern = pattern[1:]
        pattern2 = pattern2[1:]
        pattern3 = pattern3[1:]
    print("Unknown count: " + str(unknown_count))
    return prediction_output

def create_midi(prediction_output_all,mid_name):
    """ Convert the output from the prediction to notes and create a MIDI file from the notes """
    offset = 0
    output_notes = []

    for x in prediction_output_all:
        note_pattern = x[0]
        offset_pattern = x[1]
        duration_pattern = x[2]

        # Skip notes with "Unknown" values
        if note_pattern == "Unknown" or offset_pattern == "Unknown" or duration_pattern == "Unknown":
            continue

        if '.' in note_pattern or note_pattern.isdigit():
            # Chord pattern
            notes_in_chord = note_pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            duration_fraction = fractions.Fraction(duration_pattern)
            duration_float = float(duration_fraction)
            new_note.duration.quarterLength = duration_float
            output_notes.append(new_chord)
        else:
            # Single note pattern
            new_note = note.Note(note_pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            duration_fraction = fractions.Fraction(duration_pattern)
            duration_float = float(duration_fraction)
            new_note.duration.quarterLength = duration_float
            output_notes.append(new_note)
        
        offset_fraction = fractions.Fraction(offset_pattern)
        offset += float(offset_fraction)
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    output_path = "../output/"
    output_path += mid_name
    midi_stream.write('midi', fp=output_path)
    print("MIDI created!")

def generate(model_name,length,mid_name):
    """ Generate a piano midi file """
    # load the notes used to train the model
    notes, offsets, durations = get_notes(0)
    n_vocab_notes = len(set(notes))
    n_vocab_offsets = len(set(offsets))
    n_vocab_durations = len(set(durations))
    dataset_notes = MusicDataset(notes, n_vocab_notes)
    dataset_offsets = MusicDataset(offsets, n_vocab_offsets)
    dataset_durations = MusicDataset(durations, n_vocab_durations)
    
    model = create_network(n_vocab_notes, n_vocab_offsets, n_vocab_durations)
    path  = '../checkpoints/'
    path += model_name
    model.load_state_dict(torch.load(path))  # Load the trained model weights

    network_input_notes = dataset_notes.network_input
    network_input_offsets = dataset_offsets.network_input
    network_input_durations = dataset_durations.network_input
    notenames = dataset_notes.pitchnames
    offsetnames = dataset_offsets.pitchnames
    durationames = dataset_durations.pitchnames
    
    prediction_output = generate_notes(model, network_input_notes, network_input_offsets, network_input_durations, notenames, offsetnames, durationames, n_vocab_notes, n_vocab_offsets, n_vocab_durations,length)
    create_midi(prediction_output,mid_name)

if __name__ == '__main__':
    # 創建解析器對象
    parser = argparse.ArgumentParser()

    # 添加命令行參數及其縮寫
    parser.add_argument('-m', '--model_name', type=str, default='model-10.pt', help='設定模型名稱')
    parser.add_argument('-l', '--length', type=int, default=400, help='設定生成的長度')
    parser.add_argument('-n', '--mid_name', type=str, default='output.mid', help='設定生成的中間文件名稱')

    # 解析命令行參數
    args = parser.parse_args()

    # 調用 generate 函數，並將解析後的參數傳入
    generate(args.model_name, args.length, args.mid_name)
