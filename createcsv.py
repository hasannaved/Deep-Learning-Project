import csv

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
]

# Specify the file name
csv_file = 'video_data.csv'

# Write the data to the CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['video_id', 'category'])
    writer.writeheader()
    writer.writerows(video_data)

print(f'CSV file "{csv_file}" created successfully.')
