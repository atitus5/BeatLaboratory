# Dumb script so that game is playable from base directory
cd engine/
python blab.py $1 $2 || exit 1
