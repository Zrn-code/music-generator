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


def generate():
	""" Generate a piano midi file """
	#load the notes used to train the model
	notes,offsets,durations = get_notes(0)

	# Get all pitch names
	#pitchnames = sorted(set(item for item in notes))
	# Get all pitch names
	#n_vocab = len(set(notes))
	
	notenames = sorted(set(item for item in notes))
	n_vocab_notes = len(set(notes))
	network_input_notes, normalized_input_notes = prepare_sequences(notes, notenames, n_vocab_notes)
	
	offsetnames = sorted(set(item for item in offsets))
	n_vocab_offsets = len(set(offsets))
	network_input_offsets, normalized_input_offsets = prepare_sequences(offsets, offsetnames, n_vocab_offsets)
	
	durationames = sorted(set(item for item in durations))
	n_vocab_durations = len(set(durations))
	network_input_durations, normalized_input_durations = prepare_sequences(durations, durationames, n_vocab_durations)

	#model = create_network(network_input_notes, n_vocab_notes, network_input_offsets, n_vocab_offsets, network_input_durations, n_vocab_durations)
	
	model = create_network(n_vocab_notes,  n_vocab_offsets, n_vocab_durations)
	
	#network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
	#model = create_network(normalized_input, n_vocab)
	prediction_output = generate_notes(model, network_input_notes, network_input_offsets, network_input_durations, notenames, offsetnames, durationames, n_vocab_notes, n_vocab_offsets, n_vocab_durations)
	create_midi(prediction_output)

def generate_notes(model, network_input_notes, network_input_offsets, network_input_durations, notenames, offsetnames, durationames, n_vocab_notes, n_vocab_offsets, n_vocab_durations):
    """ Generate notes from the neural network based on a sequence of notes """
    start = torch.randint(0, len(network_input_notes)-1, (1,))
    start2 = torch.randint(0, len(network_input_offsets)-1, (1,))
    start3 = torch.randint(0, len(network_input_durations)-1, (1,))

    int_to_note = {number: note for number, note in enumerate(notenames)}
    int_to_offset = {number: note for number, note in enumerate(offsetnames)}
    int_to_duration = {number: note for number, note in enumerate(durationames)}

    pattern = torch.tensor(network_input_notes[start])
    pattern2 = torch.tensor(network_input_offsets[start2])
    pattern3 = torch.tensor(network_input_durations[start3])
    prediction_output = []
    unknown_count = 0
    for note_index in range(300):
        note_prediction_input = pattern.unsqueeze(0).unsqueeze(2)
        note_prediction_input = note_prediction_input / float(n_vocab_notes)

        offset_prediction_input = pattern2.unsqueeze(0).unsqueeze(2)
        offset_prediction_input = offset_prediction_input / float(n_vocab_offsets)

        duration_prediction_input = pattern3.unsqueeze(0).unsqueeze(2)
        duration_prediction_input = duration_prediction_input / float(n_vocab_durations)

        prediction = model(note_prediction_input, offset_prediction_input, duration_prediction_input)
        prediction = [F.softmax(pred, dim=1) for pred in prediction]

        try:
            index = torch.multinomial(prediction[0].reshape(-1), num_samples=1).item()
            result = int_to_note[index]
        except KeyError:
            unknown_count += 1
            continue
            result = "Unknown"

        try:
            offset = torch.multinomial(prediction[1].reshape(-1), num_samples=1).item()
            offset_result = int_to_offset[offset]
        except KeyError:
            unknown_count += 1
            continue
            offset_result = "Unknown"

        try:
            duration = torch.multinomial(prediction[2].reshape(-1), num_samples=1).item()
            duration_result = int_to_duration[duration]
        except KeyError:
            unknown_count += 1
            continue
            duration_result = "Unknown"

        print("Next note: " + str(result) + " - Duration: " + str(duration_result) + " - Offset: " + str(offset_result))

        prediction_output.append([result, offset_result, duration_result])

        pattern = torch.cat((pattern, torch.tensor([index])))
        pattern2 = torch.cat((pattern2, torch.tensor([offset])))
        pattern3 = torch.cat((pattern3, torch.tensor([duration])))
        pattern = pattern[1:]
        pattern2 = pattern2[1:]
        pattern3 = pattern3[1:]
    print("Unknown count: " + str(unknown_count))
    return prediction_output



def create_midi(prediction_output_all):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    offsets = []
    durations = []
    notes = []

    for x in prediction_output_all:
        #if x[0] == "Unknown" or x[1] == "Unknown" or x[2] == "Unknown":
        #    continue
        print(x)
        notes = np.append(notes, x[0])
        try:
            offsets = np.append(offsets, float(x[1]))
        except:
            num, denom = x[1].split('/')
            x[1] = float(num)/float(denom)
            offsets = np.append(offsets, float(x[1]))
            
        durations = np.append(durations, x[2])
	
    print("---")
    print(notes)
    print(offsets)
    print(durations)
    print("Creating Midi File...")

    # create note and chord objects based on the values generated by the model
    x = 0 # this is the counter
    for pattern in notes:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            
            try:
                new_chord.duration.quarterLength = float(durations[x])
            except:
                num, denom = durations[x].split('/')
                new_chord.duration.quarterLength = float(num)/float(denom)
            
            new_chord.offset = offset
            
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            try:
                new_note.duration.quarterLength = float(durations[x])
            except:
                num, denom = durations[x].split('/')
                new_note.duration.quarterLength = float(num)/float(denom)
            
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        try:
            offset += offsets[x]
        except:
            num, denom = offsets[x].split('/')
            offset += num/denom
        offset += 0.5
        x = x+1

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='test_output.mid')

    print("Midi created!")

if __name__ == '__main__':
	generate()
