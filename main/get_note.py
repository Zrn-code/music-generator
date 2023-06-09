import pickle
import os
import glob
from music21 import converter, instrument, note, chord

# 組合檔案的相對路徑
notes_path = './data/notes'
durations_path = './data/durations'
offsets_path = './data/offsets'

# 確定程式碼的相對位置
script_directory = os.path.dirname(os.path.abspath(__file__))

# 組合程式碼的相對路徑和檔案的相對路徑
notes_file_path = os.path.join(script_directory, notes_path)
durations_file_path = os.path.join(script_directory, durations_path)
offsets_file_path = os.path.join(script_directory, offsets_path)

def get_notes(preprocess,dataset = "classical-piano-type0"):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []
    offsets = []
    durations = []
    if preprocess == 1:
        print("Start preprocessing")
        dataset = "../dataset/" + dataset + "/*.mid"
        dataset_path = os.path.join(script_directory, dataset)
        
        for file in glob.glob(dataset_path):
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

        with open(notes_file_path, 'wb') as filepath:
            pickle.dump(notes, filepath)

        with open(durations_file_path, 'wb') as filepath:
            pickle.dump(durations, filepath)

        with open(offsets_file_path, 'wb') as filepath:
            pickle.dump(offsets, filepath)
        #print(durations)
        return notes, offsets, durations
    else:
        print("Loading preprocessed data")
        with open(notes_file_path, 'rb') as filepath:
            notes = pickle.load(filepath)
        #print(notes)
        with open(durations_file_path, 'rb') as filepath:
            durations = pickle.load(filepath)
        #print(offsets_path)
        with open(offsets_file_path, 'rb') as filepath:
            offsets = pickle.load(filepath)
        print("Preprocessed data loaded")
        return notes, offsets, durations