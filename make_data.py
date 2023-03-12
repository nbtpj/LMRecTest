# !pip install gdown
# !gdown --id 1IxAaEooIerLzKR1U0K7UL-Bz6VsaGCfW
# !gdown --id 1rur-FlPqwxrait8RPmqFkq3_AQ65o0v0
# !gdown --id 1CMYLossNv1n9Ljk3M-G8-fIwucQk-qBE
# !gzip -d Movies_and_TV.json.gz
# !gzip -d meta_Movies_and_TV.json.gz
# !ls
# !wget https://jmcauley.ucsd.edu/data/amazon_v2/categoryFiles/Movies_and_TV.json.gz --no-check-certificate
# !wget https://jmcauley.ucsd.edu/data/amazon_v2/metaFiles2/meta_Movies_and_TV.json.gz --no-check-certificate

# !gzip -d Movies_and_TV.json.gz
# !gzip -d meta_Movies_and_TV.json.gz
# !ls


"""
## Define user context prompt
___
This prompt contains the descriptions of **current** user context. They are facts (static information). In LMRec,the language model will encode this text. This definition is namely the $x(u)$.

Here, I introduce some views of such a prompt:
- bio: basic user biography, including age, gender, location, jobs
- history: rating history, including time period (e.x, 3 day ago, this user ...); movie title, movie categoris. If this number is higher than 10, top 10 most recent are adopted, followed the rest are describe statiscally (e.x, 85% movie this user saw are Crime movies, ...)
- cross-user: top most popular films and categories for each user group, by job, age, gender and location.
- cross-item: people often watch A before B, C after D, ...

In terms of causality, this information is sampled on rates having lower timestamp than that of current rate (they are UNIX timestamp, so we can compare them directly).
In test, I will combine them in multiple way.

"""

import os

os.chdir('./cache/')
from itertools import chain
from random import sample
import numpy as np
import pickle
from tqdm.auto import tqdm
from datetime import datetime
from datasets import Dataset, load_dataset, DatasetDict, load_from_disk
from typing import Iterable, Generator
from babel.dates import format_timedelta, format_datetime
from env_config import *

if not os.path.isdir('datasets.hf'):
    if not os.path.isfile('Movies_and_TV.json') or not os.path.isfile('meta_Movies_and_TV.json'):
        os.system("""
        wget https://jmcauley.ucsd.edu/data/amazon_v2/categoryFiles/Movies_and_TV.json.gz --no-check-certificate
        wget https://jmcauley.ucsd.edu/data/amazon_v2/metaFiles2/meta_Movies_and_TV.json.gz --no-check-certificate
        gzip -d Movies_and_TV.json.gz
        gzip -d meta_Movies_and_TV.json.gz
        """)
    import tensorflow_datasets as tfds

    # Construct a tf.data.Dataset
    ds = tfds.load('movielens/1m-ratings', split='train', shuffle_files=True)
    df = tfds.as_dataframe(ds)

    hf_dataset = Dataset.from_pandas(df)

    unified_dataset = DatasetDict({
        'movielens-1m-ratings': hf_dataset
    })
    ds = tfds.load('movielens/1m-movies', split='train', shuffle_files=True)
    df = tfds.as_dataframe(ds)
    meta_dataset = Dataset.from_pandas(df)
    unified_dataset['movielens-1m-movies'] = meta_dataset

    import json

    idx = -1
    with open('Movies_and_TV_platten.json', 'a', encoding='utf8') as target:
        with open('Movies_and_TV.json', 'r', encoding='utf8') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                idx += 1
                js = json.loads(line)
                for k, v in js.items():
                    if isinstance(v, dict):
                        js[k] = json.dumps(v)
                txt = json.dumps(js) if idx == 0 else '\n' + json.dumps(js)
                target.write(txt)
    unified_dataset['Movies_and_TV'] = load_dataset('json', data_files='Movies_and_TV_platten.json')['train']
    idx = -1
    with open('meta_Movies_and_TV_platten.json', 'a', encoding='utf8') as target:
        with open('meta_Movies_and_TV.json', 'r', encoding='utf8') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                idx += 1
                js = json.loads(line)
                for k, v in js.items():
                    if isinstance(v, dict) or isinstance(v, list):
                        js[k] = json.dumps(v)
                txt = json.dumps(js) if idx == 0 else '\n' + json.dumps(js)
                target.write(txt)
    unified_dataset['Movies_and_TV_meta'] = load_dataset('json', data_files='meta_Movies_and_TV_platten.json')['train']
    unified_dataset.save_to_disk('datasets.hf')
else:
    unified_dataset = load_from_disk('datasets.hf')

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
map_to_datetime = lambda x: datetime.fromtimestamp(x)

user_ids = list(set(unified_dataset['movielens-1m-ratings']['user_id']))
print('n_user: ', len(user_ids), ' n_item: ', len(unified_dataset['movielens-1m-movies']))


def film_promt(film_meta: dict) -> dict:
    title = None
    try:
        title = film_meta['movie_title'].decode('utf-8')
    except:
        pass
    categories = []
    try:
        categories = [movielen_feat_map['movie_genres'][k] for k in film_meta['movie_genres']]
    except:
        pass

    return {
        'target prompt': f'The movie named {title} is categorized as {", ".join(categories)}.{TERM_TO_ESTIMATE}'
    }


if os.path.isfile('loc_map.pick'):
    with open('loc_map.pick', 'rb') as f:
        loc_map = pickle.load(f)
else:
    from geopy.geocoders import Nominatim

    geolocator = Nominatim(user_agent="LMRec")
    zipcodes = list(set(unified_dataset['movielens-1m-ratings']['user_zip_code']))
    loc_map = {k: geolocator.geocode(k) for k in
               tqdm(zipcodes, position=0, leave=True, desc="Mapping zipcode with name")}
    with open('loc_map.pick', 'wb', ) as f:
        pickle.dump(loc_map, f)


def bio_prompt(rating: dict) -> dict:
    age = None
    try:
        age = movielen_feat_map['bucketized_user_age'][rating['bucketized_user_age']]
    except:
        pass

    job = None
    try:
        job = rating['user_occupation_text'].decode('utf-8')
    except:
        pass
    location = None
    try:
        location = loc_map[rating['user_zip_code']].address
    except:
        pass
    gender = None
    try:
        gender = movielen_feat_map['user_gender'][rating['user_gender']]
    except:
        pass
    r = f'The user is about {age} year old, and {gender} registered address of {location}.'
    if job is not None and job != 'other/not specified':
        r += f' {gender} works as {job}'
    return {
        'bio prompt': r
    }


unified_dataset['movielens-1m-ratings'] = unified_dataset['movielens-1m-ratings'].sort('timestamp')
sorted_time_stamp = np.array(unified_dataset['movielens-1m-ratings']['timestamp'])
user_occupation_label = np.array(unified_dataset['movielens-1m-ratings']['user_occupation_label'])
bucketized_user_age = np.array(unified_dataset['movielens-1m-ratings']['bucketized_user_age'])
user_gender = np.array(unified_dataset['movielens-1m-ratings']['user_gender'])
user_rating_ids = np.array(unified_dataset['movielens-1m-ratings']['user_id'])

rate2verb = {
    5: 'absolutely likes',
    4: 'likes',
    3: 'might like',
    2: 'does not like',
    1: 'hates'
}


def describe_a_rating(rating: dict, current_timestamp: int = None) -> str:
    if current_timestamp is None:
        d_time = 'in ' + str(format_datetime(map_to_datetime(rating['timestamp']), 'long'))
    else:
        d_time = format_timedelta(map_to_datetime(rating['timestamp']) - map_to_datetime(current_timestamp),
                                  add_direction=True)
    if d_time == 'in 0 seconds':
        d_time = 'recently'
    title = None
    try:
        title = rating['movie_title'].decode('utf-8')
    except:
        pass
    categories = []
    try:
        categories = [movielen_feat_map['movie_genres'][k] for k in rating['movie_genres']]
    except:
        pass
    r = f'The user {rate2verb[rating["user_rating"]]} the movie named {title}, which is categorized by {", ".join(categories)}, with the rating {rating["user_rating"]} out of 5, {d_time}.'
    return r


def describe_his(previous_ratings: dict) -> str:
    movie_cates = list(chain(*[rate['movie_genres'] for rate in previous_ratings]))
    movie_cates = np.array([movielen_feat_map['movie_genres'][k] for k in movie_cates])
    unique, counts = np.unique(movie_cates, return_counts=True)
    total = counts.sum()
    percent = counts / total * 100
    count_sort_ind = np.argsort(-counts)
    unique = unique[count_sort_ind]
    counts = counts[count_sort_ind]
    percent = np.ceil(percent[count_sort_ind])
    des = [f"""{p}% of movies this user watched categorized {u}""" if idx == 0 else f"""{p}% of them categorized {u}"""
           for idx, (u, p) in enumerate(zip(unique, percent))]
    r = f"""Generally, the user mostly watch movies having category of {', '.join(des[:5])}."""
    if len(des) > 5:
        if percent[5:].sum() <= 20:
            r += f''' In the rest of {percent[5:].sum()}%, this user watched movies categorized {', '.join(unique[5:])}.'''
        else:
            r += f''' In addition, {', '.join(des[5:])}.'''
    notseen = list(set(movielen_feat_map['movie_genres']) - set(unique))
    if 'Unknown' in notseen:
        notseen.remove('Unknown')
    if '(no genres listed)' in notseen:
        notseen.remove('(no genres listed)')
    if len(notseen) > 0:
        r += f''' Beside, this user never watch any movie categorized {', '.join(notseen)}.'''
    return r


def history_prompt(rating: dict) -> dict:
    filter_idx = np.logical_and(user_rating_ids == rating['user_id'], sorted_time_stamp < rating['timestamp'])
    filter_idx = np.argwhere(filter_idx).reshape(-1)
    previous_ratings = unified_dataset['movielens-1m-ratings'].select(filter_idx)
    t = list(previous_ratings)
    t.reverse()
    if len(previous_ratings) == 0:
        return {
            'history prompt': ''
        }
    r = 'In terms of most recent movies, ' + ' '.join([describe_a_rating(rate, rating['timestamp']) for rate in t[:10]])
    his = ''
    if len(previous_ratings) > 10:
        his = ' ' + describe_his(previous_ratings)
    return {
        'history prompt': r,
        'history overview prompt': his
    }


def get_current_trend(end_timestamp: int, period: timedelta = TRENDING_LENGTH):
    end = end_timestamp
    start = map_to_datetime(end_timestamp) - period
    start = datetime.timestamp(start)
    idx = np.argwhere(np.logical_and(sorted_time_stamp < end, sorted_time_stamp >= start))
    return unified_dataset['movielens-1m-ratings'].select(idx.reshape(-1))


def get_current_trend_in_job(rating: dict, period: timedelta = TRENDING_LENGTH):
    end = rating['timestamp']
    start = map_to_datetime(end) - period
    start = datetime.timestamp(start)
    idx = np.argwhere(np.logical_and(sorted_time_stamp < end, np.logical_and(sorted_time_stamp >= start,
                                                                             user_occupation_label == rating[
                                                                                 'user_occupation_label'])))
    return unified_dataset['movielens-1m-ratings'].select(idx.reshape(-1))


def get_current_trend_in_age(rating: dict, period: timedelta = TRENDING_LENGTH):
    end = rating['timestamp']
    start = map_to_datetime(end) - period
    start = datetime.timestamp(start)
    idx = np.argwhere(np.logical_and(sorted_time_stamp < end, np.logical_and(sorted_time_stamp >= start,
                                                                             bucketized_user_age == rating[
                                                                                 'bucketized_user_age'])))
    return unified_dataset['movielens-1m-ratings'].select(idx.reshape(-1))


def get_current_trend_in_gender(rating: dict, period: timedelta = TRENDING_LENGTH):
    end = rating['timestamp']
    start = map_to_datetime(end) - period
    start = datetime.timestamp(start)
    idx = np.argwhere(np.logical_and(sorted_time_stamp < end,
                                     np.logical_and(sorted_time_stamp >= start, user_gender == rating['user_gender'])))
    return unified_dataset['movielens-1m-ratings'].select(idx.reshape(-1))


from collections import OrderedDict

locs = []
loc_map = OrderedDict(loc_map)
for r in tqdm(loc_map):
    if loc_map[r] is not None:
        locs.append([loc_map[r].longitude, loc_map[r].latitude])
    else:
        locs.append([0, 0])
locs1, locs2, pairs = [], [], []
keys = list(loc_map.keys())
for i in range(len(locs)):
    for j in range(i):
        locs1.append(locs[i])
        locs2.append(locs[j])
        pairs.append([keys[i], keys[j]])

locs1 = np.array(locs1)
locs2 = np.array(locs2)


def spherical_dist(pos1, pos2, r=3958.75):
    pos1 = pos1 * np.pi / 180
    pos2 = pos2 * np.pi / 180
    cos_lat1 = np.cos(pos1[..., 0])
    cos_lat2 = np.cos(pos2[..., 0])
    cos_lat_d = np.cos(pos1[..., 0] - pos2[..., 0])
    cos_lon_d = np.cos(pos1[..., 1] - pos2[..., 1])
    return r * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))


dis = spherical_dist(locs1, locs2)
closed_pairs = [pairs[idx] for idx in np.argwhere(dis).reshape(-1)]
regions = {k: [k] for k in keys}
for pair in closed_pairs:
    regions[pair[0]].append(pair[1])
    regions[pair[1]].append(pair[0])

user_zip_code = unified_dataset['movielens-1m-ratings']['user_zip_code']


def get_current_trend_in_region(rating: dict, period: timedelta = TRENDING_LENGTH):
    end = rating['timestamp']
    start = map_to_datetime(end) - period
    start = datetime.timestamp(start)
    local = np.array(regions[rating['user_zip_code']])
    idx = np.argwhere(np.logical_and(sorted_time_stamp < end,
                                     np.logical_and(sorted_time_stamp >= start, np.isin(user_zip_code, local))))
    return unified_dataset['movielens-1m-ratings'].select(idx.reshape(-1))


def describe_group(current_rating: dict, previous_ratings: dict, subprompt: str) -> str:
    movie_cates = list(chain(*[rate['movie_genres'] for rate in previous_ratings]))
    movie_cates = np.array([movielen_feat_map['movie_genres'][k] for k in movie_cates])
    unique, counts = np.unique(movie_cates, return_counts=True)
    total = counts.sum()
    percent = counts / total * 100
    count_sort_ind = np.argsort(-counts)
    unique = unique[count_sort_ind]
    counts = counts[count_sort_ind]
    percent = np.ceil(percent[count_sort_ind])
    des = [f"""{p}% of movies this group watched categorized {u}""" if idx == 0 else f"""{p}% of them categorized {u}"""
           for idx, (u, p) in enumerate(zip(unique, percent))]
    r = f"""The group of {subprompt} mostly watch movies having category of {', '.join(des[:5])}."""
    if len(des) > 5:
        if percent[5:].sum() <= 40:
            r += f''' In the rest of {percent[5:].sum()}%, this group often watch movies categorized {', '.join(unique[5:])}.'''
        else:
            r += f''' In addition, {', '.join(des[5:])}.'''
    notseen = list(set(movielen_feat_map['movie_genres']) - set(unique))
    if 'Unknown' in notseen:
        notseen.remove('Unknown')
    if '(no genres listed)' in notseen:
        notseen.remove('(no genres listed)')
    if len(notseen) > 0:
        r += f''' Beside, this user group do not recently watch any movie categorized {', '.join(notseen)}.'''
    return r


def cross_user_prompt(rating,
                      period: timedelta = TRENDING_LENGTH):
    r = {}
    # group by job
    job = rating['user_occupation_text']
    if job != 'other/not specified':
        sub_prompt = f'who are {job.decode("utf-8")}'
        group_by_job = get_current_trend_in_job(rating, period)
        r_by_job = describe_group(rating, group_by_job, sub_prompt)
        r['job-group prompt'] = r_by_job
    else:
        r['job-group prompt'] = ''

    # group by age
    sub_prompt = f"who are {movielen_feat_map['bucketized_user_age'][rating['bucketized_user_age']]}"
    group_by_age = get_current_trend_in_age(rating, period)
    r_by_age = describe_group(rating, group_by_age, sub_prompt)
    r['age-group prompt'] = r_by_age

    # group by gender
    sub_prompt = f"who are {'men' if rating['user_gender'] else 'women'}"
    group_by_gender = get_current_trend_in_gender(rating, period)
    r_by_gender = describe_group(rating, group_by_gender, sub_prompt)
    r['gender-group prompt'] = r_by_gender

    # group by location
    if rating['user_zip_code'] in loc_map:
        sub_prompt = f'who are in the local region'
        group_by_region = get_current_trend_in_region(rating, period)
        r_by_region = describe_group(rating, group_by_region, sub_prompt)
        r['region-group prompt'] = r_by_region
    else:
        r['region-group prompt'] = ''
    return r


if not os.path.isdir('record_oriented.hf'):
    u_id = np.array(unified_dataset['movielens-1m-ratings']['user_id'])
    record_oriented = {
        'movie_genres_list': [],
        'movie_id_list': [],
        'movie_title_list': [],
        'timestamp_list': [],
        'user_rating_list': [],
        'user_gender': [],
        'bucketized_user_age': [],
        'user_id': [],
        'user_occupation_label': [],
        'user_occupation_text': [],
        'user_zip_code': []
    }
    for each_id in tqdm(np.unique(u_id).reshape(-1)):
        user_rate_list = unified_dataset['movielens-1m-ratings'].select(np.argwhere(u_id == each_id).reshape(-1))
        for each_feat in user_rate_list.column_names:
            if each_feat in record_oriented:
                record_oriented[each_feat].append(user_rate_list[0][each_feat])
            elif each_feat + '_list' in record_oriented:
                record_oriented[each_feat + '_list'].append(user_rate_list[each_feat])
    record_oriented = Dataset.from_dict(record_oriented)
    record_oriented.save_to_disk('record_oriented.hf')
else:
    record_oriented = load_from_disk('record_oriented.hf')


def continuous_double_iter(iterable: Iterable, num_stop_iter: int = None) -> Generator:
    x = iter(iterable)
    st = next(x)
    try:
        while True and (num_stop_iter is None or num_stop_iter > 0):
            if num_stop_iter is not None:
                num_stop_iter -= 1
            next_v = next(x)
            yield (st, next_v)
            st = next_v
    except StopIteration as e:
        return


def unmesh_pair(a, b) -> Generator:
    for a_ in a:
        for b_ in b:
            yield (a_, b_)


if not os.path.isfile('supporting_sequences.pick'):
    import tqdm

    movie_ids, movie_id_timestamps, movie_id_ratings = [], [], []
    movie_cates, movie_cate_timestamps, movie_cate_ratings = [], [], []
    for cate_list, movie_list, rating_list, time_stamp in tqdm.tqdm(zip(record_oriented['movie_genres_list'],
                                                                        record_oriented['movie_id_list'],
                                                                        record_oriented['user_rating_list'],
                                                                        record_oriented['timestamp_list'])):
        for meshedpair, timestamppair, ratingpair in zip(continuous_double_iter(cate_list),
                                                         continuous_double_iter(time_stamp),
                                                         continuous_double_iter(rating_list)):
            unmesh_pairs = list(unmesh_pair(*meshedpair))
            movie_cates.extend(unmesh_pairs)
            movie_cate_timestamps.extend([timestamppair] * len(unmesh_pairs))
            movie_cate_ratings.extend([ratingpair] * len(unmesh_pairs))
        for idpair, timestamppair, ratingpair in zip(continuous_double_iter(movie_list),
                                                     continuous_double_iter(time_stamp),
                                                     continuous_double_iter(rating_list)):
            movie_ids.append(idpair)
            movie_id_timestamps.append(timestamppair)
            movie_id_ratings.append(ratingpair)
    with open('supporting_sequences.pick', 'wb') as f:
        pickle.dump(
            (movie_ids, movie_id_timestamps, movie_id_ratings, movie_cates, movie_cate_timestamps, movie_cate_ratings),
            f)
else:
    with open('supporting_sequences.pick', 'rb') as f:
        (movie_ids, movie_id_timestamps, movie_id_ratings, movie_cates, movie_cate_timestamps,
         movie_cate_ratings) = pickle.load(f)
assert len(movie_cates) == len(movie_cate_ratings) == len(movie_cate_timestamps)
assert len(movie_ids) == len(movie_id_timestamps) == len(movie_id_ratings)

movie_ids = np.array(movie_ids)
movie_id_timestamps = np.array(movie_id_timestamps)
movie_id_ratings = np.array(movie_id_ratings)
movie_cates = np.array(movie_cates)
movie_cate_timestamps = np.array(movie_cate_timestamps)
movie_cate_ratings = np.array(movie_cate_ratings)

maptotext = {k: v for k, v in zip(unified_dataset['movielens-1m-movies']['movie_id'],
                                  unified_dataset['movielens-1m-movies']['movie_title'])}


def movie_id_pair2text(pair):
    return maptotext[pair[0]].decode('utf-8'), maptotext[pair[1]].decode('utf-8')


def cate_id_pair2text(pair):
    return movielen_feat_map['movie_genres'][pair[0]], movielen_feat_map['movie_genres'][pair[1]]


def statistic_pair_trend(pairs, rating_pairs):
    vals, inverse, counts = np.unique(pairs, axis=0, return_inverse=True, return_counts=True)
    args = np.flip(np.argsort(counts))
    vals = vals[args]
    counts = counts[args]
    total = counts.sum()
    assert rating_pairs.shape[0] == pairs.shape[0]
    for pair, n_sample in zip(vals, counts):
        popularity = n_sample / total * 100
        a_rating = np.mean(rating_pairs[np.argwhere((pairs == pair).all(axis=1)).reshape(-1), 0])
        b_rating = np.mean(rating_pairs[np.argwhere((pairs == pair).all(axis=1)).reshape(-1), 1])
        yield pair, popularity, a_rating, b_rating, n_sample


def next_movie_prompt(rating: dict) -> dict:
    """
    Note: I categorize this prompt is a fact, not an inference,
    because we can extract it easily from current dataset without modeling
    """
    # filter by time
    idxs = np.argwhere(movie_id_timestamps[..., -1] < rating['timestamp']).reshape(-1)
    rating_trends = movie_id_ratings[idxs, ...]
    _movie_ids = movie_ids[idxs, ...]
    # filter by movie id and divide into 2 pattern: A->B and B->A for rating comments, with A is current film of this rating
    A2B_idx = np.argwhere(np.isin(_movie_ids[..., 0], [rating['movie_id']])).reshape(-1)
    A2B_rating_trends = rating_trends[A2B_idx, ...]
    A2B_ids = _movie_ids[A2B_idx, ...]

    B2A_idx = np.argwhere(np.isin(_movie_ids[..., -1], [rating['movie_id']])).reshape(-1)
    B2A_rating_trends = rating_trends[B2A_idx, ...]
    B2A_ids = _movie_ids[B2A_idx, ...]

    # create description

    prompts = []
    for pair, popularity, a_rating, b_rating, n_sample in statistic_pair_trend(A2B_ids, A2B_rating_trends):
        if popularity < INFORMATIVE_P_VALUE or n_sample < INFORMATIVE_SAMPLE_SIZE:
            break
        else:
            names = movie_id_pair2text(pair)
            prompts.append(
                f"""{n_sample} people, equal to {popularity:.0f} percent of who have rated {names[0]}, then continuously rated {names[1]}. For these people, the averaged rating for {names[0]} is {a_rating:.1f} while this for {names[1]} is {b_rating:.1f}""")

    for pair, popularity, a_rating, b_rating, n_sample in statistic_pair_trend(B2A_ids, B2A_rating_trends):
        if popularity < INFORMATIVE_P_VALUE or n_sample < INFORMATIVE_SAMPLE_SIZE:
            break
        else:
            names = movie_id_pair2text(pair)
            prompts.append(
                f"""{n_sample} people, equal to {popularity:.0f} who would rate {names[1]}, have rated {names[0]}. For these people, the averaged rating for {names[0]} is {a_rating:.1f} while this for {names[1]} is {b_rating:.1f}""")
    return {
        'cross-movie prompt': ' '.join(prompts[:5])
    }


def next_cate_prompt(rating: dict) -> dict:
    """
    Note: I categorize this prompt is a fact, not an inference,
    because we can extract it easily from current dataset without modeling
    """
    if 17 in rating['movie_genres']:
        rating['movie_genres'].remove(17)

    if 20 in rating['movie_genres']:
        rating['movie_genres'].remove(20)

    # filter by time
    idxs = np.argwhere(movie_cate_timestamps[..., -1] < rating['timestamp']).reshape(-1)
    rating_trends = movie_cate_ratings[idxs, ...]
    _movie_cates = movie_cates[idxs, ...]
    # filter by movie id and divide into 2 pattern: A->B and B->A for rating comments, with A is current film of this rating
    A2B_idx = np.argwhere(np.isin(_movie_cates[..., 0], rating['movie_genres'])).reshape(-1)
    A2B_rating_trends = rating_trends[A2B_idx, ...]
    A2B_ids = _movie_cates[A2B_idx, ...]

    B2A_idx = np.argwhere(np.isin(_movie_cates[..., -1], rating['movie_genres'])).reshape(-1)
    B2A_rating_trends = rating_trends[B2A_idx, ...]
    B2A_ids = _movie_cates[B2A_idx, ...]

    # create description
    prompts = []
    for pair, popularity, a_rating, b_rating, n_sample in statistic_pair_trend(A2B_ids, A2B_rating_trends):
        if popularity < INFORMATIVE_P_VALUE or n_sample < INFORMATIVE_SAMPLE_SIZE:
            break
        else:
            names = cate_id_pair2text(pair)
            prompts.append(
                f"""{n_sample} people, equal to {popularity:.0f} percent who has rated a film categorized {names[0]}, then continuously rated a film categorized {names[1]}. For these people, the averaged rating for {names[0]} is {a_rating:.1f} while this for {names[1]} is {b_rating:.1f}""")

    for pair, popularity, a_rating, b_rating, n_sample in statistic_pair_trend(B2A_ids, B2A_rating_trends):
        if popularity < INFORMATIVE_P_VALUE or n_sample < INFORMATIVE_SAMPLE_SIZE:
            break
        else:
            names = cate_id_pair2text(pair)
            prompts.append(
                f"""{n_sample} people, equal to {popularity:.0f} percent who would rate a film categorized {names[1]}, has rated a film categorized {names[0]}. For these people, the averaged rating for {names[0]} is {a_rating:.1f} while this for {names[1]} is {b_rating:.1f}""")
    return {
        'cross-cate prompt': ' '.join(prompts[:5])
    }


prompt_types = [
    bio_prompt,
    history_prompt,
    cross_user_prompt,
    next_movie_prompt,
    next_cate_prompt,
]


def combine_prompt(rating):
    d = dict()
    for strategy in prompt_types:
        d.update(strategy(rating))
    return d


total_samples = len(unified_dataset['movielens-1m-ratings'])
_SAMPLE_SIZE = 10000
sample_idxs = sample(list(range(total_samples)), _SAMPLE_SIZE, )

if not os.path.isdir(f'sampled_prompt_movilens_dataset.hf'):
    shard = unified_dataset['movielens-1m-ratings'].select(sample_idxs, keep_in_memory=True)
    sampled_prompt_movilens_dataset = shard.map(combine_prompt,
                                                num_proc=10,
                                                cache_file_name='sampled_prompt_movilens_dataset.cache')
    sampled_prompt_movilens_dataset.save_to_disk(f'sampled_prompt_movilens_dataset.hf')
else:
    from datasets import load_from_disk

    sampled_prompt_movilens_dataset = load_from_disk('sampled_prompt_movilens_dataset.hf')
