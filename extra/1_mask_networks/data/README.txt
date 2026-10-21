All audio in this folder comes from the LibriSpeech ASR corpus
(V. Panayotov, G. Chen, D. Povey, S. Khudanpur, ICASSP 2015), https://www.openslr.org/12 ,
licensed under Creative Commons Attribution 4.0 (CC BY 4.0). Converted to 16-bit, 16 kHz.

train/              117 clips for TRAINING: LibriSpeech dev-clean, 3 utterances from each of 39 speakers
                    (every dev-clean speaker except 1272), each cut to its first 5 s.
                    The list is in train_list.txt.

speech_sample.wav   The TEST sentence, the same one as in Week 2:
                    dev-clean 1272-135031-0007 "He eats and sleeps very steadily, replied the new king."
                    Speaker 1272 is not in train/, so the networks never hear this voice while training.

noise_babble.flac   TEST noise of a kind the networks never train on: six people talking at once
                    (LibriSpeech test-clean, 3 women and 3 men, first 10 s each, summed at equal level).
                    The list is in babble_list.txt.
