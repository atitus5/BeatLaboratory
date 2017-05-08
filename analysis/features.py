import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

if len(sys.argv) != 2:
    print "Usage: python spectrogram.py <log file>"
    sys.exit(1)

# Read in spectrogram data
log_filename = sys.argv[1]
spectrogram = pd.read_csv(log_filename).as_matrix().T

plt.pcolor(spectrogram, cmap="gnuplot2")
plt.show()
