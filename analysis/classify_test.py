import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

if len(sys.argv) != 2:
    print "Usage: python classify_test.py <log file>"
    sys.exit(1)

# Read in spectrogram data
log_filename = sys.argv[1]
spectrogram = pd.read_csv(log_filename).as_matrix().T
labels = {
    39: "kick",
    66: "snare",
    119: "snare",
    162: "kick",
    175: "snare",
}

kBassThreshold = 0.6
kDebounceBackoff = 0.95
kSnareIndices = (12, 75)

# Track last time we classified, so we can debounce the signal
last_classified = 1024     # Arbitrarily high number so that we start off with no debouncing

for frame_idx in xrange(spectrogram.shape[1]):
    debounced_bass = (1.0 - (kDebounceBackoff ** last_classified)) * spectrogram[1, frame_idx]
    if debounced_bass >= kBassThreshold:
        snare_sum = sum(spectrogram[kSnareIndices[0]:kSnareIndices[1], frame_idx])
        print "Event at frame %d" % frame_idx
        print "  Debounced bass: %.3f" % debounced_bass
        print "  Snare sum: %.3f" % snare_sum
        print "  Bass/snare: %.3f" % (debounced_bass / snare_sum)
        last_classified = 0
    else:
        last_classified += 1
