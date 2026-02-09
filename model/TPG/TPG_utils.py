from typing import Union, Tuple
import math

from nltk import ngrams
import numpy as np

EarthRadius = 6378137
MinLatitude = -85.05112878
MaxLatitude = 85.05112878
MinLongitude = -180
MaxLongitude = 180


def clip(n: float, min_value: float, max_value: float) -> float:
    return min(max(n, min_value), max_value)


def map_size(level_of_detail: int) -> int:
    return 256 << level_of_detail


def latlon2pxy(
    latitude: float, longitude: float, level_of_detail: int, swin_type: str
) -> Tuple[list[int], list[int]]:
    latitude = clip(latitude, MinLatitude, MaxLatitude)
    longitude = clip(longitude, MinLongitude, MaxLongitude)

    x = (longitude + 180) / 360
    sin_latitude = math.sin(latitude * math.pi / 180)
    y = 0.5 - math.log((1 + sin_latitude) / (1 - sin_latitude)) / (4 * math.pi)

    size = map_size(level_of_detail)
    pixel_x = int(clip(x * size + 0.5, 0, size - 1))
    pixel_y = int(clip(y * size + 0.5, 0, size - 1))
    interval = 64
    if swin_type == "cross":
        return [pixel_x, pixel_x, pixel_x, pixel_x + interval, pixel_x - interval], [
            pixel_y,
            pixel_y - interval,
            pixel_y + interval,
            pixel_y,
            pixel_y,
        ]
    elif swin_type == "grid":
        return [
            pixel_x,
            pixel_x + interval,
            pixel_x - interval,
            pixel_x + interval,
            pixel_x - interval,
        ], [
            pixel_y,
            pixel_y - interval,
            pixel_y - interval,
            pixel_y + interval,
            pixel_y + interval,
        ]
    elif swin_type == "mix":
        return [
            pixel_x,
            pixel_x,
            pixel_x,
            pixel_x + interval,
            pixel_x - interval,
            pixel_x + interval,
            pixel_x - interval,
            pixel_x + interval,
            pixel_x - interval,
        ], [
            pixel_y,
            pixel_y - interval,
            pixel_y + interval,
            pixel_y,
            pixel_y,
            pixel_y - interval,
            pixel_y - interval,
            pixel_y + interval,
            pixel_y + interval,
        ]
    else:
        raise ValueError("swin type {} is not available!".format(swin_type))


def txy2quadkey(
    tile_x: list[int], tile_y: list[int], level_of_detail: int
) -> list[str]:
    quadkey_list = []

    for x, y in zip(tile_x, tile_y):
        quadkey = []
        for i in range(level_of_detail, 0, -1):
            digit = 0
            mask = 1 << (i - 1)
            if (x & mask) != 0:
                digit += 1
            if (y & mask) != 0:
                digit += 2
            quadkey.append(str(digit))
        quadkey_list.append("".join(quadkey))
    return quadkey_list


def pxy2txy(pixel_x: list[int], pixel_y: list[int]) -> Tuple[list[int], list[int]]:
    tile_x = []
    tile_y = []
    for x, y in zip(pixel_x, pixel_y):
        tile_x.append(x // 256)
        tile_y.append(y // 256)
    return tile_x, tile_y


def latlon2quadkey(lat: float, lon: float, level: int, swin_type: str) -> list[str]:
    """
    Convert longitude/latitude to a Bing Maps quadkey string.
    """
    pixel_x, pixel_y = latlon2pxy(lat, lon, level, swin_type)
    tile_x, tile_y = pxy2txy(pixel_x, pixel_y)
    return txy2quadkey(tile_x, tile_y, level)


def build_region_id(
    poi_id_seqs: list[int],
    lat_seqs: list[float],
    lon_seqs: list[float],
    length: int = 9,
) -> Tuple[dict[int, list[list[str]]], list[list[list[str]]]]:
    region_quadkey_bigrams_map = {}
    all_quadkey_bigrams: list[list[list[str]]] = []

    for _ in range(length):
        all_quadkey_bigrams.append([])
    for poi_id, lat, lon in zip(poi_id_seqs, lat_seqs, lon_seqs):
        regions = latlon2quadkey(float(lat), float(lon), 17, "mix")
        region_quadkey_bigrams = []
        for idx, region_quadkey in enumerate(regions):
            region_quadkey_bigram = " ".join(
                ["".join(x) for x in ngrams(region_quadkey, 6)]
            )
            region_quadkey_bigram_list = region_quadkey_bigram.split()
            all_quadkey_bigrams[idx].append(region_quadkey_bigram_list)
            region_quadkey_bigrams.append(region_quadkey_bigram_list)
        region_quadkey_bigrams_map[poi_id] = region_quadkey_bigrams
    # add padding
    region_quadkey_bigrams_map[0] = region_quadkey_bigrams_map[1]
    return region_quadkey_bigrams_map, all_quadkey_bigrams


class QuadkeyField:
    def __init__(self):
        self.vocab = {}
        self.idx_to_token = {}
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"

    def build_vocab(self, data: list[list[list[str]]]) -> None:
        all_tokens = set()
        for seq in data:
            for item in seq:
                if isinstance(item, str):
                    tokens = item.split()
                else:
                    tokens = item
                all_tokens.update(tokens)

        self.vocab = {self.pad_token: 0, self.unk_token: 1}
        for idx, token in enumerate(sorted(all_tokens), 2):
            self.vocab[token] = idx

        self.idx_to_token = {v: k for k, v in self.vocab.items()}

    def encode(self, text: Union[str, list[str]]) -> list:
        if isinstance(text, str):
            tokens = text.split()
        else:
            tokens = text
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]

    def decode(self, indices: list[int]) -> list[str]:
        return [self.idx_to_token.get(idx, self.unk_token) for idx in indices]

    def numericalize(self, data: Union[list[list[str]]]) -> np.ndarray:
        if isinstance(data, list):
            return np.array([self.encode(item) for item in data], dtype=np.int64)
        else:
            return np.array(self.encode(data), dtype=np.int64)
