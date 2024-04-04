import sqlite3
import numpy as np
import re
from youtube_transcript_api import YouTubeTranscriptApi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
import pandas as pd
# Download NLTK resources (only download once)
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Initialize WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_text(transcript):
    """
    Preprocesses the transcript text.
    """
    # Initialize a list to store preprocessed tokens
    preprocessed_tokens = []

    # Retrieve the English stop words
    stop_words = set(stopwords.words('english'))

    # Iterate through each segment of the transcript
    for segment in transcript:
        # Convert segment text to lowercase
        segment_text = segment['text'].lower()
        # Remove non-alphanumeric characters and tokenize text
        tokens = word_tokenize(re.sub(r'[^a-zA-Z0-9\s]', '', segment_text))
        # Remove stop words and lemmatize tokens
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        # Add preprocessed tokens to the list
        preprocessed_tokens.extend(tokens)

    # Join tokens back into a single string
    preprocessed_text = ' '.join(preprocessed_tokens)
    return preprocessed_text

def fetch_and_store_transcripts(video_ids, category):
    """
    Fetches and stores transcripts for the specified videos in the SQLite database.
    """
    # Connect to SQLite database
    conn = sqlite3.connect('youtube_transcripts.db')
    c = conn.cursor()
    
    # Create a table to store transcripts if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS transcripts
                 (video_id TEXT, transcript TEXT, category INTEGER)''')

    for video_id in video_ids:
        try:
            # Retrieve transcript for the video
            transcript = YouTubeTranscriptApi.get_transcript(video_id)

            # Preprocess transcript text
            preprocessed_transcript = preprocess_text(transcript)

            # Store transcript in the database
            c.execute("INSERT INTO transcripts VALUES (?, ?, ?)", (video_id, preprocessed_transcript, category))
            conn.commit()

            print(f"Transcript for video {video_id} stored successfully.")
        except Exception as e:
            print(f"Error processing video {video_id}: {e}")

    # Close database connection
    conn.close()

# Read video data from CSV file
video_data_df = pd.read_csv('video_data.csv')

# Fetch and store preprocessed transcripts for the specified videos
for index, row in video_data_df.iterrows():
    video_id = row['video_id']
    category = row['category']
    fetch_and_store_transcripts([video_id], category)

# Load data from the SQLite database
conn = sqlite3.connect('youtube_transcripts.db')
c = conn.cursor()
c.execute("SELECT transcript, category FROM transcripts")
data = c.fetchall()
conn.close()

# Split data into transcripts and corresponding categories
transcripts, categories = zip(*data)

# Train Word2Vec model to convert text data into numerical representations
tokenized_transcripts = [transcript.split() for transcript in transcripts]
word2vec_model = Word2Vec(tokenized_transcripts, vector_size=100, window=5, min_count=1, workers=4)

# Convert text data to numerical representations using Word2Vec embeddings
X = []
for transcript in tokenized_transcripts:
    embedding = np.zeros((len(transcript), 100))
    for i, word in enumerate(transcript):
        if word in word2vec_model.wv:
            embedding[i] = word2vec_model.wv[word]
    X.append(embedding)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, categories, test_size=0.2, random_state=42)

# Pad sequences to have the same length
max_length = max(len(seq) for seq in X_train)
X_train_padded = pad_sequences(X_train, maxlen=max_length, padding='post', dtype='float32')
X_test_padded = pad_sequences(X_test, maxlen=max_length, padding='post', dtype='float32')

# Define the RNN model
model = Sequential()
model.add(Bidirectional(LSTM(128)))
model.add(Dense(3, activation='softmax'))

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_padded, np.array(y_train), epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_padded, np.array(y_test))
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Save the trained model
model.save('transcript_classification_model.keras')

# Load the saved model (for demonstration)
# loaded_model = keras.models.load_model('transcript_classification_model.keras')

#///HyperParameter tuning///
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier

# # Define a function to create the RNN model with custom hyperparameters
# def create_rnn_model(learning_rate=0.001, dropout_rate=0.2, num_lstm_units=128):
#     model = Sequential()
#     model.add(Bidirectional(LSTM(num_lstm_units)))
#     model.add(Dropout(dropout_rate))
#     model.add(Dense(3, activation='softmax'))
#     optimizer = Adam(learning_rate=learning_rate)
#     model.compile(optimizer=optimizer,
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model

# # Define a function to create the KerasClassifier with custom hyperparameters
# def create_keras_classifier(learning_rate=0.001, dropout_rate=0.2, num_lstm_units=128):
#     return KerasClassifier(build_fn=create_rnn_model, epochs=10, batch_size=32, verbose=0,
#                            learning_rate=learning_rate, dropout_rate=dropout_rate,
#                            num_lstm_units=num_lstm_units)

# # Wrap the Keras model so it can be used with GridSearchCV
# rnn_model = create_keras_classifier()

# # Define the hyperparameter grid
# param_grid = {
#     'learning_rate': [0.001, 0.01, 0.1],
#     'dropout_rate': [0.2, 0.3, 0.4],
#     'num_lstm_units': [64, 128, 256],
# }

# # Perform grid search
# grid_search = GridSearchCV(estimator=rnn_model, param_grid=param_grid, cv=3)
# grid_result = grid_search.fit(X_train_padded, np.array(y_train))

# # Print the best hyperparameters
# print("Best Hyperparameters: ", grid_result.best_params_)

# # Evaluate the best model on the test set
# best_model = grid_result.best_estimator_
# test_loss, test_accuracy = best_model.score(X_test_padded, np.array(y_test))
# print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

