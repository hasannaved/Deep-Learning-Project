from youtube_transcript_api import YouTubeTranscriptApi
# retrieve the available transcripts
# transcript_list = YouTubeTranscriptApi.list_transcripts('iwh8XbZk-zQ','bIDKhZ_4jLQ&ab_channel=DudePerfect')
                                                        # ,'ll8bMh1I7oU','tJPX_RkjYtc','qUUloBe5vEo','dwV04XuiWq4','g9G44Az00l0','JErXWwcVA6Q','viQC-6xoJ3E','3TJP5a3pBME','_XFzT9GMmw8','aUDgaN6iHFc','uWFv1vuPtwo','l2s_mknWn-w','aJq9bmwv0CI','mkbtYFxDnWo','KK21LIfAF6I','p-84FvmpeEw','nj87csGC6c8','iPMaoRmCUoQ&ab_channel=KieranBrown')

# for transcript in transcript_list:
#     # fetch the actual transcript data
#     print(transcript.fetch())

import sqlite3
from youtube_transcript_api import YouTubeTranscriptApi

# Define a function to fetch and store transcripts
def fetch_and_store_transcripts(video_ids, category):
    # Connect to SQLite database
    conn = sqlite3.connect('youtube_transcripts.db')
    c = conn.cursor()
    
    # Create a table to store transcripts if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS transcripts
                 (video_id TEXT, title TEXT, transcript TEXT, category TEXT)''')

    for video_id in video_ids:
        try:
            # Retrieve transcript for the video
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = transcript_list.find_transcript(['en'])

            # Fetch video title
            video_info = YouTubeTranscriptApi.get_transcript(video_id)
            title = video_info.title

            # Store transcript in the database
            c.execute("INSERT INTO transcripts VALUES (?, ?, ?, ?)", (video_id, title, transcript, category))
            conn.commit()

            print(f"Transcript for video {video_id} stored successfully.")
        except Exception as e:
            print(f"Error processing video {video_id}: {e}")

    # Close database connection
    conn.close()

# Define the video IDs and their corresponding categories
video_data = [
    {'video_id': 'iwh8XbZk-zQ', 'category': 'sports'},
    {'video_id': 'bIDKhZ_4jLQ&ab_channel=DudePerfect', 'category': 'sports'},
    {'video_id': 'll8bMh1I7oU', 'category': 'sports'},
    # {'video_id': 'tJPX_RkjYtc', 'category': 'sports'},
    # {'video_id': 'qUUloBe5vEo', 'category': 'sports'},
    # {'video_id': 'dwV04XuiWq4', 'category': 'sports'},
    # {'video_id': 'g9G44Az00l0', 'category': 'sports'},
    # {'video_id': 'JErXWwcVA6Q', 'category': 'sports'},
    # {'video_id': 'viQC-6xoJ3E', 'category': 'sports'},
    # {'video_id': '3TJP5a3pBME', 'category': 'sports'},
    # {'video_id': '_XFzT9GMmw8', 'category': 'sports'},
    # {'video_id': 'aUDgaN6iHFc', 'category': 'sports'},
    # {'video_id': 'uWFv1vuPtwo', 'category': 'sports'},
    # {'video_id': 'l2s_mknWn-w', 'category': 'sports'},
    # {'video_id': 'aJq9bmwv0CI', 'category': 'sports'},
    # {'video_id': 'mkbtYFxDnWo', 'category': 'sports'},
    # {'video_id': 'KK21LIfAF6I', 'category': 'sports'},
    # {'video_id': 'p-84FvmpeEw', 'category': 'sports'},
    # {'video_id': 'nj87csGC6c8', 'category': 'sports'},
    # {'video_id': 'iPMaoRmCUoQ&ab_channel=KieranBrown', 'category': 'sports'},
  
    # Add more videos and categories as needed
]

# Fetch and store transcripts for the specified videos
for video_info in video_data:
    video_id = video_info['video_id']
    category = video_info['category']
    fetch_and_store_transcripts([video_id], category)

# Connect to the database
conn = sqlite3.connect('youtube_transcripts.db')
c = conn.cursor()

# Execute a query to fetch all rows from the transcripts table
c.execute("SELECT * FROM transcripts")
rows = c.fetchall()

# Display the fetched rows
for row in rows:
    print(row)

# Close the database connection
conn.close()