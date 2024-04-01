from youtube_transcript_api import YouTubeTranscriptApi
import sqlite3
import numpy as np
import re
from youtube_transcript_api import YouTubeTranscriptApi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Bidirectional
from keras.preprocessing.sequence import pad_sequences

# Download NLTK resources
import nltk
#/////////////////////////DOWNLOAD NLTK ONCE//////////////////////////
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Initialize WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_text(transcript):
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

# Define a function to fetch and store transcripts
def fetch_and_store_transcripts(video_ids, category):
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


# Define the video IDs and their corresponding categories
category_mapping = {'sports': 0, 'travel': 1, 'art': 2}  

video_data = [
    {'video_id': 'iwh8XbZk-zQ', 'category': 0},
    {'video_id': 'bIDKhZ_4jLQ&ab_channel=DudePerfect', 'category': 0},
    {'video_id': 'tJPX_RkjYtc', 'category': 0},
    {'video_id': 'qUUloBe5vEo', 'category': 0},
    {'video_id': 'dwV04XuiWq4', 'category': 0},
    {'video_id': 'JErXWwcVA6Q', 'category': 0},
    {'video_id': 'viQC-6xoJ3E', 'category': 0},
    {'video_id': '3TJP5a3pBME', 'category': 0},
    {'video_id': '_XFzT9GMmw8', 'category': 0},
    {'video_id': 'aUDgaN6iHFc', 'category': 0},
    {'video_id': 'uWFv1vuPtwo', 'category': 0},
    {'video_id': 'l2s_mknWn-w', 'category': 0},
    {'video_id': 'aJq9bmwv0CI', 'category': 0},
    {'video_id': 'p-84FvmpeEw', 'category': 0},
    {'video_id': 'nj87csGC6c8', 'category': 0},
    {'video_id': 'iPMaoRmCUoQ&ab_channel=KieranBrown', 'category': 0},
    {'video_id': 'z5qHEhnabnE', 'category': 2},
    {'video_id': 'c9ngFduPffI', 'category': 2},
    {'video_id': 'D2M9W7qTAWg', 'category': 2},
    {'video_id': 'Lt2RJepYz2k', 'category': 2},
    {'video_id': '7n9d-yBpxy4', 'category': 2},
    {'video_id': 'phQ0Hfbnz2s', 'category': 2},
    {'video_id': 'EwRySI6xvWI', 'category': 2},
    {'video_id': 'EwGTHz4isu8', 'category': 2},
    {'video_id': 'csvyIVIaol4', 'category': 2},
    {'video_id': 'aBSirlqP2uw', 'category': 2},
    {'video_id': 'HjY1y0Kixj0', 'category': 2},
    {'video_id': 'WcjvEVkq03o', 'category': 2},
    {'video_id': 'LhkbIj7CgT0', 'category': 2},
    {'video_id': 'lGgkvndtYh0', 'category': 2},
    {'video_id': 'IlFXmMriFmA', 'category': 2},
    {'video_id': 'YcXpAHVAxwY', 'category': 2}
    # Add more videos and categories as needed
    ]


# Fetch and store preprocessed transcripts for the specified videos
for video_info in video_data:
    video_id = video_info['video_id']
    category = video_info['category']
    fetch_and_store_transcripts([video_id], category)

# Step 1: Load data from the SQLite database
conn = sqlite3.connect('youtube_transcripts.db')
c = conn.cursor()
c.execute("SELECT transcript, category FROM transcripts")
data = c.fetchall()
conn.close()

# Split data into transcripts and corresponding categories
transcripts, categories = zip(*data)

# Step 2: Train Word2Vec model to convert text data into numerical representations
tokenized_transcripts = [transcript.split() for transcript in transcripts]
word2vec_model = Word2Vec(tokenized_transcripts, vector_size=100, window=5, min_count=1, workers=4)

# Step 3: Convert text data to numerical representations using Word2Vec embeddings
X = []
for transcript in tokenized_transcripts:
    embedding = np.zeros((len(transcript), 100))
    for i, word in enumerate(transcript):
        if word in word2vec_model.wv:
            embedding[i] = word2vec_model.wv[word]
    X.append(embedding)

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, categories, test_size=0.2, random_state=42)


from keras.preprocessing.sequence import pad_sequences
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
