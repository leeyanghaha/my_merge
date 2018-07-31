import json
import operator
from collections import Counter

import utils.spacy_utils as su
import utils.pattern_utils as pu
import utils.tweet_keys as tk
import sqlite3
from geocoder.google import GoogleResult


class CursorFactory:
    cursor = None
    
    @staticmethod
    def get_cursor():
        if CursorFactory.cursor is None:
            print('making connection to database')
            conn = sqlite3.connect("/home/nfs/cdong/tw/src/models/geo/geo_location")
            CursorFactory.cursor = conn.cursor()
        return CursorFactory.cursor
    
    @staticmethod
    def get_geo_by_loc_name(loc_name):
        cursor = CursorFactory.get_cursor()
        query = "SELECT i2g.GEO FROM nameToId AS n2i, idToGeo AS i2g WHERE n2i.NAME=:place AND n2i.ID=i2g.ID"
        cursor.execute(query, {"place": loc_name})
        geo_result = cursor.fetchone()
        if not geo_result:
            return None
        else:
            json_obj = json.loads(geo_result[0])
            return GoogleResult(json_obj)


def extract_geo_freq_list(twarr, docarr):
    coord_list = get_coordinates(twarr)
    loc_freq_list = get_loc_freq_list(docarr)
    geo_freq_list = get_geo_freq_list(loc_freq_list, coord_list)
    return geo_freq_list


def get_coordinates(twarr):
    k_coord = tk.key_coordinates
    coord_list = list()
    for tw in twarr:
        if (k_coord in tw) and (tw[k_coord] is not None) and (k_coord in tw[k_coord]):
            coord_list.append(tw[k_coord][k_coord])
    return coord_list


def get_loc_freq_list(docarr):
    loc_freq_counter = Counter()
    for doc in docarr:
        ents = [e for e in doc.ents if (e.label_ in su.LABEL_LOCATION and len(e.text) > 1)]
        locs = [pu.capitalize(e.text) for e in ents]
        loc_freq_counter += Counter(set(locs))
    if len(loc_freq_counter) == 0:
        return list()
    loc_freq_list = loc_freq_counter.most_common()
    top_loc, top_freq = loc_freq_list[0]
    v_loc_freq_list = [(loc, freq) for loc, freq in loc_freq_list if freq > (top_freq / 3)]
    return v_loc_freq_list


def get_geo_freq_list(loc_freq_list, coordinates):
    geo_freq_list = _get_geo_freq_list(loc_freq_list)
    geo_freq_list = _reevaluate_with_coordinates(geo_freq_list, coordinates)
    geo_freq_list = _determine_index_order(geo_freq_list)
    return geo_freq_list


def _get_geo_freq_list(loc_freq_list):
    geo_freq_list = list()
    for loc, freq in loc_freq_list:
        geo = CursorFactory.get_geo_by_loc_name(loc)
        if geo:
            geo_freq_list.append((geo, freq))
    return geo_freq_list


def _reevaluate_with_coordinates(geo_freq_list, coordinate_list):
    if not geo_freq_list:
        return list()
    # for each geo, count the coordinates that falls in its region
    coordinates_match = list()
    for idx, (geo, freq) in enumerate(geo_freq_list):
        match_num = 0
        for coord_lng, coord_lat in coordinate_list:
            if geo.west < coord_lng < geo.east and geo.south < coord_lat < geo.north:
                match_num += 1
        coordinates_match.append(match_num)
    origin_max_match = geo_freq_list[0][1]
    coord_max_match = max(coordinates_match)
    if coord_max_match == 0:
        return geo_freq_list
    new_geo_list = list()
    for idx, (geo, freq) in enumerate(geo_freq_list):
        # 0.42 is approximately 3/7
        score = freq + 0.42 * origin_max_match * coordinates_match[idx] / coord_max_match
        new_geo_list.append((geo, freq, score))
    new_geo_list = sorted(new_geo_list, key=operator.itemgetter(2), reverse=True)
    new_geo_freq_list = [(geo, freq) for geo, freq, new_freq in new_geo_list]
    return new_geo_freq_list


def _determine_index_order(geo_list, index1=0, index2=1):
    if index1 >= len(geo_list) or index2 >= len(geo_list) or index1 == index2:
        return geo_list
    geo1_quality = geo_list[index1][0].quality
    geo2_quality = geo_list[index2][0].quality
    if geo1_quality == 'country':
        if geo2_quality != 'country':
            temp = geo_list[index1]
            geo_list[index1] = geo_list[index2]
            geo_list[index2] = temp
            return geo_list
        else:
            index2 += 1
            return _determine_index_order(geo_list, index1, index2)
    else:
        return geo_list


if __name__ == '__main__':
    cursor = CursorFactory.get_cursor()
    loc_name = "Asia"
    query = "SELECT i2g.GEO FROM nameToId AS n2i, idToGeo AS i2g WHERE n2i.NAME=:place AND n2i.ID=i2g.ID"
    cursor.execute(query, {"place": loc_name})
    geo_result = GoogleResult(json.loads(cursor.fetchone()[0]))
    print(geo_result.quality)
    print(geo_result.city)
    print(geo_result.country_long)
    print(geo_result.bbox)
    print(geo_result.raw)
    print(geo_result.fieldnames)
    exit()
#
#     sql_create_table_name_to_id = """ CREATE TABLE IF NOT EXISTS nameToId (NAME text PRIMARY KEY,ID text);"""
#     sql_create_table_id_to_Geo = """ CREATE TABLE IF NOT EXISTS idToGeo (ID text PRIMARY KEY,GEO text NOT NULL); """
#     if conn:
#         print('Geo database opened successfully')
#     c.execute(sql_create_table_name_to_id)
#     c.execute(sql_create_table_id_to_Geo)
#     conn.commit()
