from typing import Tuple, Iterator

import math
from itertools import tee
from itertools import product
import pandas as pd

import torch
from torch import nn

EarthRadius = 6378137
MinLatitude = -85.05112878
MaxLatitude = 85.05112878
Minlnggitude = -180
Maxlnggitude = 180


def clip(n: float, minValue: float, maxValue: float) -> float:
    return min(max(n, minValue), maxValue)


def map_size(levelOfDetail: int) -> int:
    return 256 << levelOfDetail


def latlng2pxy(
    latitude: float, lnggitude: float, levelOfDetail: int
) -> Tuple[int, int]:
    latitude = clip(latitude, MinLatitude, MaxLatitude)
    lnggitude = clip(lnggitude, Minlnggitude, Maxlnggitude)

    x = (lnggitude + 180) / 360
    sinLatitude = math.sin(latitude * math.pi / 180)
    y = 0.5 - math.log((1 + sinLatitude) / (1 - sinLatitude)) / (4 * math.pi)

    mapSize = map_size(levelOfDetail)
    pixelX = int(clip(x * mapSize + 0.5, 0, mapSize - 1))
    pixelY = int(clip(y * mapSize + 0.5, 0, mapSize - 1))
    return pixelX, pixelY


def txy2quadkey(tileX: int, tileY: int, levelOfDetail: int) -> str:
    quadKey = []
    for i in range(levelOfDetail, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (tileX & mask) != 0:
            digit += 1
        if (tileY & mask) != 0:
            digit += 2
        quadKey.append(str(digit))

    return "".join(quadKey)


def pxy2txy(pixelX: int, pixelY: int) -> Tuple[int, int]:
    tileX = pixelX // 256
    tileY = pixelY // 256
    return tileX, tileY


def latlng2quadkey(lat: float, lng: float, level: int) -> str:
    pixelX, pixelY = latlng2pxy(lat, lng, level)
    tileX, tileY = pxy2txy(pixelX, pixelY)
    return txy2quadkey(tileX, tileY, level)


def ngrams(sequences: str, n: int, **kwargs) -> zip:
    """
    Return the ngrams generated from a sequence of items, as an iterator.
    """
    sequence = pad_sequence(sequences, **kwargs)

    # Creates the sliding window, of n no. of items.
    # `iterables` is a tuple of iterables where each iterable is a window of n items.
    iterables = tee(sequence, n)

    for i, sub_iterable in enumerate(iterables):  # For each window,
        for _ in range(i):  # iterate through every order of ngrams
            next(sub_iterable, None)  # generate the ngrams within the window.
    return zip(*iterables)  # Unpack and flattens the iterables.


def pad_sequence(sequence: str) -> Iterator[str]:
    """
    Returns a padded sequence of items before ngram extraction.
    """
    s = iter(sequence)
    return s


def rotate(
    head: torch.Tensor, relation: torch.Tensor, hidden: int, device: torch.device
) -> torch.Tensor:
    re_head, im_head = torch.chunk(head, 2, dim=-1)

    # Make phases of relations uniformly distributed in [-pi, pi]
    embedding_range = nn.Parameter(
        torch.Tensor([(24.0 + 2.0) / hidden]), requires_grad=False
    ).to(device)

    phase_relation = relation / (embedding_range / torch.pi)

    re_relation = torch.cos(phase_relation)
    im_relation = torch.sin(phase_relation)

    re_score = re_head * re_relation - im_head * im_relation
    im_score = re_head * im_relation + im_head * re_relation

    score = torch.cat([re_score, im_score], dim=-1)
    return score


def rotate_batch(
    head: torch.Tensor, relation: torch.Tensor, hidden: int, device: torch.device
) -> torch.Tensor:
    pi = 3.14159265358979323846

    re_head, im_head = torch.chunk(head, 2, dim=2)

    # Make phases of relations uniformly distributed in [-pi, pi]
    embedding_range = nn.Parameter(
        torch.Tensor([(24.0 + 2.0) / hidden]), requires_grad=False
    ).to(device)

    phase_relation = relation / (embedding_range / pi)

    re_relation = torch.cos(phase_relation)
    im_relation = torch.sin(phase_relation)

    re_score = re_head * re_relation - im_head * im_relation
    im_score = re_head * im_relation + im_head * re_relation

    score = torch.cat([re_score, im_score], dim=2)
    return score


def get_all_permutations_dict(length: int) -> dict[str, int]:
    characters = ["0", "1", "2", "3"]

    all_permutations = ["".join(p) for p in product(characters, repeat=length)]

    premutation_dict = dict(zip(all_permutations, range(len(all_permutations))))

    return premutation_dict


def get_norm_time96(time: pd.Timestamp) -> float:
    time = pd.to_datetime(time, unit='s')
    hour = time.hour
    minute = time.minute

    ans = minute // 15 + 4 * hour

    return ans / 96


def get_day_norm7(time: pd.Timestamp) -> float:
    time = pd.to_datetime(time, unit='s')
    day_number = time.dayofweek
    return day_number / 7


def get_time_slot_id(time: pd.Timestamp) -> int:
    time = pd.to_datetime(time, unit='s')
    minute = time.minute
    hour = time.hour
    day_number = time.dayofweek

    if minute <= 30:
        ans = 2 * hour
    else:
        ans = 2 * hour + 1

    if day_number >= 5:
        return ans + 48
    else:
        return ans


def get_ngrams_of_quadkey(
    quadkey: str, n: int, permutations_dict: dict[str, int]
) -> list[int]:
    region_quadkey_bigram = " ".join(["".join(x) for x in ngrams(quadkey, n)])
    region_quadkey_bigram_list = region_quadkey_bigram.split()
    ret = [permutations_dict[each] for each in region_quadkey_bigram_list]
    return ret


def get_quad_keys(
    lat: float,
    lon: float,
    permutations_dict: dict[str, int],
    quadkey_len: int = 25,
    ngrams: int = 6,
) -> list[int]:
    quadkey = latlng2quadkey(lat, lon, quadkey_len)
    quadkeys = get_ngrams_of_quadkey(quadkey, ngrams, permutations_dict)
    return quadkeys
