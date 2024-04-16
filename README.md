The main file is Project.py, which uses the youtube_transcript API to download the transcripts of the videos IDs listed in video_data.csv. The transcripts are then preprocessed, tokenized, and using word2vec transformed into a suitable format. The dataset is then split into test and train and an RNN Sequential LSTM model is trained on it. 

The youtube descriptions before they are preprocessed and tokenized are stored in thr youtube_transcripts.db database. 

The file hyperparameter tuning is used to optimize the the model and find the best combination of learning rate, dropout rate, and LSTM units. 