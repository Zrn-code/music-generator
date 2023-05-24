import pickle
import glob
from music21 import converter, instrument, note, chord

def get_notes(preprocess):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []
    offsets = []
    durations = []
    if preprocess == 1:
        print("Start preprocessing")
        for file in glob.glob("../classical-piano-type0/*.mid"):
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
        #print(durations)
        return notes, offsets, durations
    else:
        with open('data/notes', 'rb') as filepath:
            notes = pickle.load(filepath)

        with open('data/durations', 'rb') as filepath:
            durations = pickle.load(filepath)

        with open('data/offsets', 'rb') as filepath:
            offsets = pickle.load(filepath)
        #print(durations)
        return notes, offsets, durations