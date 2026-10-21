All audio in this folder comes from the LibriSpeech ASR corpus
(V. Panayotov, G. Chen, D. Povey, S. Khudanpur, ICASSP 2015), https://www.openslr.org/12 ,
licensed under Creative Commons Attribution 4.0 (CC BY 4.0). Converted to 16-bit, 16 kHz.

train/              117 clips for TRAINING, the same as Week 3: LibriSpeech dev-clean, 3 utterances
                    from each of 39 speakers (every dev-clean speaker except 1272), each cut to its
                    first 5 s. The list is in train_list.txt.

test/               NEW this week: 20 sentences for TESTING, LibriSpeech test-clean, 10 women and
                    10 men, one sentence per speaker, each cut to its first 4 s. None of these
                    speakers is in train/ or in the babble. The list is in test_list.txt.

speech_sample.wav   The Week 2 / Week 3 test sentence (dev-clean 1272-135031-0007, speaker 1272).
                    Lab 1 uses it to reproduce the Week 3 result.

noise_babble.flac   TEST noise, the same as Week 3: six people talking at once (LibriSpeech
                    test-clean, 3 women and 3 men, first 10 s each, summed at equal level).
                    The list is in babble_list.txt. Never used for training.

Who is where (no voice appears in more than one place):

    train/  (and the training babble made from it in Lab 3)   39 dev-clean speakers
    speech_sample.wav                                          dev-clean speaker 1272
    test/                                                      20 test-clean speakers
    noise_babble.flac                                          6 other test-clean speakers
