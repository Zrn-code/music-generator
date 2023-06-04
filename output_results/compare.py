import music21
import os
script_directory = os.path.dirname(os.path.abspath(__file__))



# 載入音樂檔案
score1 = music21.converter.parse(os.path.join(script_directory, "real_music.mid"))
score2 = music21.converter.parse(os.path.join(script_directory, "output.mid"))

import matplotlib.pyplot as plt

# 獲取所有音符和和弦
notes_and_chords = score1.flat.notesAndRests.stream()
notes_and_chords2 = score2.flat.notesAndRests.stream()

# 提取音符的offset和音高
offsets = []
pitches = []
for element in notes_and_chords:
    if element.isNote:
        offsets.append(element.offset)
        pitches.append(element.pitch.ps)
    elif element.isChord:
        chord_pitches = [p.ps for p in element.pitches]
        avg_pitch = sum(chord_pitches) / len(chord_pitches)
        offsets.append(element.offset)
        pitches.append(avg_pitch)



# 繪製散點圖
plt.scatter(offsets, pitches, marker='o', color='b')

# 提取音符的offset和音高
offsets = []
pitches = []
for element in notes_and_chords2:
    if element.isNote:
        offsets.append(element.offset)
        pitches.append(element.pitch.ps)
    elif element.isChord:
        chord_pitches = [p.ps for p in element.pitches]
        avg_pitch = sum(chord_pitches) / len(chord_pitches)
        offsets.append(element.offset)
        pitches.append(avg_pitch)

plt.scatter(offsets, pitches, marker='o', color='r')


# 設定圖表標籤
plt.xlabel('Offset')
plt.ylabel('Pitch')
plt.title('Music Structure')

# 顯示圖表
plt.show()
