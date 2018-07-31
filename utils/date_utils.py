import datetime
import time
import re

strptime = datetime.datetime.strptime


month_dict = {'Jan': '1', 'Feb': '2', 'Mar': '3', 'Apr': '4', 'May': '5', 'Jun': '6',
               'Jul': '7', 'Aug': '8', 'Sep': '9', 'Oct': '10', 'Nov': '11', 'Dec': '12'}


def parse_created_at(created_at_str):
    """
    :param created_at_str:
    :return:  array of string, indicating the time components
    """
    time_split = created_at_str.strip().split(' ')
    year = time_split[5]
    month_of_year = month_dict[time_split[1]]
    date_of_month = time_split[2]
    day_of_week = time_split[0]
    time_of_day = time_split[3]
    return year, month_of_year, date_of_month, day_of_week, time_of_day


def timestamp_of_string(time_str):
    try:
        d = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        return int(time.mktime(d.timetuple()))
    except:
        try:
            d = datetime.datetime.strptime(time_str, "%Y-%m-%d")
            return int(time.mktime(d.timetuple()))
        except:
            raise ValueError('Incorrect format: {}'.format(time_str))


def get_timestamp_form_created_at(created_at_str):
    # "created_at": "Thu Jul 14 23:33:57 +0000 2016" / "Wed Feb 17 17:14:14 +0000 2016"
    year, month_of_year, date_of_month, day_of_week, time_of_day = parse_created_at(created_at_str)
    return timestamp_of_string('-'.join([year, month_of_year, date_of_month]) + ' ' + time_of_day)


def get_timestamp_form_string(time_string):
    return timestamp_of_string('-'.join(re.split('[^\d]', time_string)))


def normalize_ymdh(ymdh_str_arr):
    ymdh_arr = list()
    for item in ymdh_str_arr:
        if type(item) is str:
            ymdh_arr.append(item)
        else:
            raise TypeError("Elements of ymdh array is not of valid type str.")
    return ymdh_arr


def is_target_ymdh(tweet_ymdh, start_ymdh, end_ymdh):
    """ymdh resembles ['201X', '0X', '2X', '1X'] ,with element of type string"""
    if start_ymdh is [] and end_ymdh is []:
        # No ymdh limit given, taken as all dates.
        return True
    if (start_ymdh is not [] or end_ymdh is not []) and tweet_ymdh is []:
        # Once start or end hes been given, then tweet should be provided as well.
        return False
    ds = dt = de = None
    
    format_str_arr = ['%Y', '%m', '%d', '%H']
    normed_tw_ymdh = normalize_ymdh(tweet_ymdh)
    ele_num = len(normed_tw_ymdh)
    if tweet_ymdh is not []:
        dt = strptime('-'.join(normed_tw_ymdh), '-'.join(format_str_arr[:ele_num]))
    if start_ymdh is not []:
        ds = strptime('-'.join(normalize_ymdh(start_ymdh)[:ele_num]), '-'.join(format_str_arr[:ele_num]))
    if end_ymdh is not []:
        de = strptime('-'.join(normalize_ymdh(end_ymdh)[:ele_num]), '-'.join(format_str_arr[:ele_num]))
    if ds and de and (ds - de).days > 0:
        raise ValueError('Ending date earlier than beginning date.')
    is_t_lt_s = (dt - ds).days >= 0 if ds else True
    is_e_lt_t = (de - dt).days >= 0 if de else True
    return is_t_lt_s and is_e_lt_t


def compare_ymdh(ymdh1, ymdh2):
    """ymdh resembles ['201X', '0X', '2X', '1X'] ,with element of type string"""
    format_str_arr = ['%Y', '%m', '%d', '%H']
    time1 = datetime.datetime.strptime('-'.join(ymdh1), '-'.join(format_str_arr[:len(ymdh1)]))
    time2 = datetime.datetime.strptime('-'.join(ymdh2), '-'.join(format_str_arr[:len(ymdh2)]))
    return (time1 - time2).days >= 0
