import pitchtypes as pt
import re
class Chord:

    ROOTS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    QUALITIES = ['major', 'minor', 'sus', 'unknown']
    
    ENHARMONIC_MAP = {
        'Cb': 'B',  'B#': 'C',
        'Db': 'C#', 'Eb': 'D#',
        'Fb': 'E',  'E#': 'F',
        'Gb': 'F#', 'Ab': 'G#',
        'Bb': 'A#',
        # Allow canonical too
        'C': 'C', 'C#': 'C#', 'D': 'D', 'D#': 'D#', 'E': 'E',
        'F': 'F', 'F#': 'F#', 'G': 'G', 'G#': 'G#', 'A': 'A', 'A#': 'A#', 'B': 'B'
    }


    def __init__(self, chord_str):
            self.label = chord_str

            # Match root with any number of accidentals: A-G followed by # or b
            match = re.match(r'^([A-Ga-g][#b]*)', chord_str)
            if match:
                raw_root = match.group(1).capitalize()
            else:
                raise ValueError(f"Invalid chord string: {chord_str}")

            try:
                # Normalize enharmonically
                enharmonic = str(pt.EnharmonicPitchClass(str(pt.SpelledPitchClass(raw_root))))
                self.root = pt.SpelledPitchClass(enharmonic)
            except Exception as e:
                raise ValueError(f"Could not parse root {raw_root}: {e}")

            root_str_length = len(match.group(1))
            remainder = chord_str[root_str_length:]

            if 'sus' in remainder:
                self.quality = 'sus'
            elif 'm' in remainder:
                self.quality = 'minor'
            elif '^' in remainder or remainder == '' or remainder[0] in ['6', '7']:
                self.quality = 'major'
            else:
                self.quality = 'unknown'

    def distance_to(self, chord_other):
        spelled_interval = str(self.root.interval_to(chord_other.root))
        return int(pt.EnharmonicIntervalClass(spelled_interval))

    def distance_from(self, chord_other):
        spelled_interval = str(self.root.interval_from(chord_other.root))
        return int(pt.EnharmonicIntervalClass(spelled_interval))
    
    def __repr__(self):
        return f"Chord(root={self.root}, quality={self.quality})"
    
    @staticmethod
    def encode_chord(chord_str):
        def one_hot(value, vocab):
            vec = [0] * len(vocab)
            if value in vocab:
                vec[vocab.index(value)] = 1
            return vec

        # Parse root and normalize
        if len(chord_str) >= 2 and chord_str[1] in ['#', 'b']:
            root = chord_str[0:2]
            root_str_len = 2
        else:
            root = chord_str[0]
            root_str_len = 1

        root = Chord.ENHARMONIC_MAP.get(root, 'unknown')

        remainder = chord_str[root_str_len:]

        if 'sus' in remainder:
            quality = 'sus'
        elif 'm' in remainder:
            quality = 'minor'
        elif '^' in remainder or remainder == '' or remainder[0] in ['6', '7']:
            quality = 'major'
        else:
            quality = 'unknown'

        root_vec = one_hot(root, Chord.ROOTS)
        quality_vec = one_hot(quality, Chord.QUALITIES)
        return root_vec + quality_vec  # vector of length 18