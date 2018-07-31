import datetime
import random
import traceback
from collections import Counter

import pytz
import timezonefinder
from dateutil import parser
from prettytable import PrettyTable
from sutime import SUTime

import utils.tweet_keys as tk

tf = timezonefinder.TimezoneFinder()
jar_path = "/home/nfs/cdong/tw/src/tools/CoreNLP/"
sutime = None


def get_sutime():
    global sutime
    if sutime is None:
        sutime = SUTime(jars=jar_path, mark_time_ranges=True)
    return sutime


'''
输入: tweet list, 推测出的事件的GoogleResult
输出: text_times, utc_time
utc_time: 推断出的事件时间 type：datetime
text_times: [(推测时间，推文文本，时间词，推文创建时间，utc_offset)]
'''


def get_text_time(twarr, relevant_geo=None, sutime_obj=None):
    if sutime_obj is None:
        sutime_obj = get_sutime()
    geo_timezone = get_geo_timezone(relevant_geo)
    text_times, extract_times = list(), list()
    for tw in twarr:
        tw_created_at, tw_text = tw[tk.key_created_at], tw[tk.key_text]
        try:
            tweet_time = datetime.datetime.strptime(tw_created_at, '%a %b %d %H:%M:%S %z %Y')
            parsed_time_list = sutime_obj.parse(tw_text, reference_date=tweet_time.date().isoformat())
        except:
            continue
        if not parsed_time_list:
            continue
        if tk.key_user in tw and tk.key_utc_offset in tw[tk.key_user] \
                and tw[tk.key_user][tk.key_utc_offset] is not None:
            utc_offset = tw[tk.key_user][tk.key_utc_offset]
        else:
            utc_offset = None
        ''' time zone: get utc_offset from user object in tweet metadata to: 文本中的时区 '''
        try:
            for parsed_time in parsed_time_list:
                
                parse_datetime = _get_parsed_datetime(parsed_time, tweet_time, geo_timezone)
                if parse_datetime:
                    # print("extract_time_info line 98", parse_datetime)
                    local_time = geo_timezone.localize(parse_datetime.replace(tzinfo=None))
                    utc_time = local_time.astimezone(tz=datetime.timezone.utc)
                    text_times.append(
                        (utc_time, tw_text, parsed_time['text'], tweet_time,
                         utc_offset / 3600 if utc_offset else 'None'))
                    extract_times.append((parse_datetime, parsed_time, tweet_time))
        
        except:
            # traceback.print_exc()
            continue
    dt = predict_most_common(extract_times)
    if dt:
        local_time = geo_timezone.localize(dt.replace(tzinfo=None))
        utc_time = local_time.astimezone(tz=datetime.timezone.utc)
    else:
        discarded, utc_time = get_earlist_latest_post_time(twarr)
    return text_times, utc_time


'''
从timex3 value 提取 datetime
能提取到则返回 datetime类型
不能提取到则返回 None
'''


def _get_parsed_datetime(parsed_time, tweet_time, geo_timezone):
    time_type, value, time_text = parsed_time['type'], parsed_time['value'], parsed_time['text']
    time_text_lower = time_text.lower()
    parse_datetime = None
    '''
    handle case: time type is 'Time'
    '''
    if time_type == 'TIME':
        if len(value) > 5 and value[-3] == ':' and value[-6] == 'T':
            parse_datetime = parser.parse(value)
            # print("template T:1400", parse_datetime)
        elif len(value) > 3 and value[-3:].isalpha():
            indicate_word = value[-3:]
            # night time TNI TAF TEV
            random_minute = random.randint(0, 59)
            if indicate_word == 'TNI':
                random_hour = random.randint(22, 23)
                parse_datetime = parser.parse(value[:-3] + ' {}:{}'.format(random_hour, random_minute))
            elif indicate_word == 'TAF':
                random_hour = random.randint(13, 15)
                parse_datetime = parser.parse(value[:-3] + ' {}:{}'.format(random_hour, random_minute))
            elif indicate_word == 'TEV':
                random_hour = random.randint(19, 21)
                parse_datetime = parser.parse(value[:-3] + ' {}:{}'.format(random_hour, random_minute))
        # 处理没有 am pm 的情况
        if parse_datetime:
            if 0 < parse_datetime.hour <= 12:
                if 'am' not in time_text_lower and 'pm' not in time_text_lower and ':' in time_text:
                    try:
                        # tweet_time = datetime.datetime.strptime(
                        #     tw_created_at, '%a %b %d %H:%M:%S %z %Y')
                        tmp_datetime = parse_datetime + datetime.timedelta(hours=12)
                        bear_margin = 60 * 60
                        if -bear_margin < \
                                (geo_timezone.localize(tmp_datetime.replace(tzinfo=None)).astimezone(datetime.timezone.utc) - tweet_time).total_seconds() \
                                < bear_margin:
                            parse_datetime = tmp_datetime
                    except:
                        # traceback.print_exc()
                        pass
    elif time_type == 'DATE':
        if 'w' in value.lower() or value == 'FUTURE_REF':
            # week weekend & FUTURE_REF not in consideration
            return None
        # present_ref use as date or time ??
        if value == "PRESENT_REF":
            if time_text_lower == 'now':
                # word "Now" usually dose not mean the current time
                return None
            parse_datetime = tweet_time
        elif len(value) > 7:
            parse_datetime = parser.parse(value)
    
    return parse_datetime


def predict_most_common(extract_times):
    # dates=[(parse_datetime, time_phase, tweet), (parse_datetime, time_phase, tweet), ...]
    dates = [time_tuple for time_tuple in extract_times if time_tuple[1]['type'] == 'DATE']
    times = [time_tuple for time_tuple in extract_times if time_tuple[1]['type'] == 'TIME']
    date_counter = Counter()
    for date in dates:
        parse_datetime, time_phase = date[0], date[1]
        if time_phase['value'] == 'PRESENT_REF':
            continue
        dt = parse_datetime.date()
        date_counter[dt] += 1
    time_counter = Counter()
    for time in times:
        parse_datetime = time[0]
        tm = _get_time_in_intervals(parse_datetime).time()
        time_counter[tm] += 1
    date_list = date_counter.most_common()
    time_list = time_counter.most_common()
    
    if len(date_list) > 0:
        case = 1 if len(time_list) > 0 else 2
    else:
        case = 3 if len(time_list) > 0 else 4
    
    if case == 1:
        # date_list non-empty, time_list non-empty, and fetch both top
        return _predict_with_date_and_time(date_list=date_list, time_list=time_list)
    elif case == 2:
        # date_list non-empty, time_list empty
        return _predict_with_date_no_time(date_list=date_list, dates=dates)
    elif case == 3:
        # date_list empty, time_list non-empty
        return _predict_no_date_with_time(time_list=time_list, times=times)
    elif case == 4:
        return None


def _predict_with_date_and_time(date_list, time_list):
    date = date_list[0][0]
    time = time_list[0][0]
    return datetime.datetime(
        year=date.year, month=date.month, day=date.day, hour=time.hour, minute=time.minute,
        second=time.second, microsecond=time.microsecond, tzinfo=time.tzinfo)


def _predict_with_date_no_time(date_list, dates):
    date = date_list[0][0]
    earliest_time = None
    for d_date in dates:
        try:
            parse_datetime, tw_created_at = d_date[0], d_date[0]
            if parse_datetime.date() == date:
                if earliest_time:
                    tmp_time = tw_created_at
                    if (tmp_time - earliest_time).total_seconds() < 0:
                        earliest_time = tmp_time
                else:
                    earliest_time = tw_created_at
        except:
            # traceback.print_exc()
            continue
    time = earliest_time.time()
    
    return datetime.datetime(
        year=date.year, month=date.month, day=date.day, hour=time.hour, minute=time.minute,
        second=time.second, microsecond=time.microsecond, tzinfo=time.tzinfo)


def _predict_no_date_with_time(time_list, times):
    time = time_list[0][0]
    find_date = Counter()
    for d_time in times:
        parse_datetime = d_time[0]
        if _get_time_in_intervals(parse_datetime).time() == time:
            tmp_date = parse_datetime.date()
            find_date[tmp_date] += 1
    find_date_list = find_date.most_common()
    date = find_date_list[0][0]
    return datetime.datetime(
        year=date.year, month=date.month, day=date.day, hour=time.hour, minute=time.minute,
        second=time.second, microsecond=time.microsecond, tzinfo=time.tzinfo)


def _get_time_in_intervals(dtime):
    time_interval = 10
    minute_minus = dtime.minute % time_interval
    dtime = dtime - datetime.timedelta(minutes=minute_minus, seconds=dtime.second)
    return dtime


def get_geo_timezone(geo):
    # geo is an instance of GoogleResult
    utc = pytz.timezone("utc")
    if not geo:
        return utc
    geo_lat, geo_lng = geo.lat, geo.lng
    timezone_str = tf.certain_timezone_at(lat=geo_lat, lng=geo_lng)
    if not timezone_str:
        # print("get_geo_timezone Could not determine the time zone")
        timezone = utc
    else:
        # print("get_geo_timezone", timezone_str)
        from pytz import UnknownTimeZoneError
        try:
            timezone = pytz.timezone(timezone_str)
        except UnknownTimeZoneError:
            return utc
    return timezone


def get_earlist_latest_post_time(twarr):
    if not twarr:
        return None, None
    earliest_time = latest_time = None
    for tw in twarr:
        try:
            tw_created_at = tw[tk.key_created_at]
            tweet_time = datetime.datetime.strptime(tw_created_at, '%a %b %d %H:%M:%S %z %Y')
            if earliest_time is None or tweet_time < earliest_time:
                earliest_time = tweet_time
            if latest_time is None or tweet_time > latest_time:
                latest_time = tweet_time
        except:
            # traceback.print_exc()
            continue
    
    # now = datetime.datetime.now(tz=datetime.timezone.utc)
    # if earliest_time is None:
    #     earliest_time = now
    # if latest_time is None:
    #     latest_time = now
    return earliest_time, latest_time


# if __name__ == '__main__':
#     test_case = u'I need a desk for tomorrow from 2pm to 3pm'
#     print(json.dumps(sutime.parse(test_case), sort_keys=True, indent=4))
if __name__ == '__main__':
    import utils.function_utils as fu
    import utils.file_iterator as fi
    
    base = '/home/nfs/cdong/tw/seeding/Terrorist/queried/positive'
    pos_files = fi.listchildren(base, concat=True)
    for file in pos_files:
        _twarr = fu.load_array(file)

        _text_times, _utc_time = get_text_time(_twarr)
        _earliest_time, _latest_time = get_earlist_latest_post_time(_twarr)
        
        print(_earliest_time.isoformat())
        print(_latest_time.isoformat())
        print(_utc_time.isoformat())
        
        _table = PrettyTable(["推测时间", "推文文本", "时间词", "推文创建时间", "utc_offset"])
        _table.padding_width = 1
        for _time in _text_times:
            _table.add_row(_time)
        print(_table)
