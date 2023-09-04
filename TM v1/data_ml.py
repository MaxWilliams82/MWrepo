#les 4 premiers chiffres sont les pixels, le 5ème le target (paysage ou non)
#ordre des pixels de l'image:
#12
#34
import random

data = [
[1, 1, 0, 0, 1],
[0, 0, 1, 1, 1],
[0, 0, 0, 0, 0],
[1, 0, 0, 0, 0],
[0, 1, 0, 0, 0],
[0, 0, 1, 0, 0],
[0, 0, 0, 1, 0],
[0, 1, 1, 0, 0],
[1, 0, 0, 1, 0],
[1, 1, 1, 0, 0],
[1, 1, 0, 1, 0],
[1, 0, 1, 1, 0],
[0, 1, 1, 1, 0],
[1, 1, 1, 1, 0],
[0, 1, 0, 1, 0],
[1, 0, 1, 0, 0],
]


nb_data = len(data)
input_size = len(data[0]) - 1 #car dans cet exemple le target est aussi dans le ligne, mais ce n'est pas un input

