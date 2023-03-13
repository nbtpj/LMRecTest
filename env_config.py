from datetime import timedelta
TERM_TO_ESTIMATE = " This user wants to see this movie."
DEEP_MODEL_BATCH_SIZE = 8
DEFAULT_REGION_RADIUS_IN_MILE = 30
INFORMATIVE_SAMPLE_SIZE = 30
INFORMATIVE_P_VALUE = 5
TRENDING_LENGTH = timedelta(days = 10)
RATINGS = [
 "This user rate this movie 1 out of 5.",
 "This user rate this movie 2 out of 5.",
 "This user rate this movie 3 out of 5.",
 "This user rate this movie 4 out of 5.",
 "This user rate this movie 5 out of 5.",
]

movielen_feat_map = {
    'movie_genres': [
        'Action',
        'Adventure',
        'Animation',
        'Children',
        'Comedy',
        'Crime',
        'Documentary',
        'Drama',
        'Fantasy',
        'Film-Noir',
        'Horror',
        'IMAX',
        'Musical',
        'Mystery',
        'Romance',
        'Sci-Fi',
        'Thriller',
        'Unknown',
        'War',
        'Western',
        '(no genres listed)',
    ],
    'user_occupation_label': [
        'academic/educator',
        'artist',
        'clerical/admin',
        'customer service',
        'doctor/health care',
        'entertainment',
        'executive/managerial',
        'farmer',
        'homemaker',
        'lawyer',
        'librarian',
        'other/not specified',
        'programmer',
        'retired',
        'sales/marketing',
        'scientist',
        'self-employed',
        'student',
        'technician/engineer',
        'tradesman/craftsman',
        'unemployed',
        'writer',
    ],
    'bucketized_user_age': {
        1: "under 18",
        18: "from 18 to 24",
        25: "from 25 to 34",
        35: "from 35 to 44",
        45: "from 45 to 49",
        50: "from 50 to 55",
        56: "over 56",
    },
    'user_gender': {
        True: 'he',
        False: 'she'
    }
}


import torch 

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model_paralell = device == 'cuda:0' and torch.cuda.device_count() > 1