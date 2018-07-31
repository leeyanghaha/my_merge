import utils.tweet_keys as tk
import utils.pattern_utils as pu


FILTER_LEVEL_LOW = 'low'
FILTER_LEVEL_HIGH = 'high'
FILTER_LEVEL_NONE = 'none'

low_tw_attr = {'created_at', 'timestamp_ms', 'id', 'text', 'place', 'user',
               'retweet_count', 'favorite_count', 'entities', 'source',
               'filter_level', 'truncated', 'is_quote_status',
               'in_reply_to_status_id', 'in_reply_to_user_id'}
low_user_attr = {'id', 'created_at', 'time_zone', 'location', 'favourites_count',
                 'followers_count', 'friends_count', 'listed_count', 'statuses_count',
                 'contributors_enabled', 'protected', 'is_translator',
                 'description', 'verified'}

high_tw_attr = {'created_at', 'timestamp_ms', 'id', 'text', 'place', 'user',
                'retweet_count', 'favorite_count', 'in_reply_to_status_id', 'in_reply_to_user_id'}
high_user_attr = {'id', 'created_at', 'time_zone', 'location', 'favourites_count',
                  'followers_count', 'friends_count', 'listed_count', 'statuses_count', 'verified'}

attr_dict = {FILTER_LEVEL_LOW: (low_tw_attr, low_user_attr),
             FILTER_LEVEL_HIGH: (high_tw_attr, high_user_attr),
             FILTER_LEVEL_NONE: (None, None)}


def filter_twarr(twarr, filter_level=FILTER_LEVEL_LOW):
    """ may result in change in the length of twarr """
    tw_attrs, usr_attrs = attr_dict[filter_level]
    removal_idx = list()
    for idx, tw in enumerate(twarr):
        if tk.key_lang not in tw or not tw[tk.key_lang] == 'en' or tk.key_text not in tw:
            removal_idx.append(idx)
            continue
        """ filter attributes """
        tw[tk.key_user] = filter_attribute(tw[tk.key_user], usr_attrs)
        """ filter text """
        tw[tk.key_orgntext] = tw[tk.key_text]
        normalized_text = filter_text(tw[tk.key_text])
        if len(pu.tokenize(r'[a-zA-Z_\-\']{3,}', normalized_text)) <= 5:
            removal_idx.append(idx)
            continue
        tw[tk.key_text] = normalized_text
    
    for idx in removal_idx[::-1]:
        del twarr[idx]
    return twarr


def twarr_dup_id(twarr, get_id=lambda tw: tw.get(tk.key_id)):
    id_set, dup_idx_list = set(), list()
    for _idx, _tw in enumerate(twarr):
        tw_id = get_id(_tw)
        if tw_id not in id_set:
            id_set.add(tw_id)
        else:
            dup_idx_list.append(_idx)
    return dup_idx_list


# def filter_twarr_attr(twarr, attr_filter=None,
#                       tw_cond=lambda tw: tk.key_lang in tw and tw[tk.key_lang] == 'en' and tk.key_text in tw,
#                       filter_level=FILTER_LEVEL_LOW):
#     if attr_filter is not None:
#         for tw in twarr:
#             if tw_cond(tw):
#                 attr_filter(tw)
#         return twarr
#     else:
#         res_twarr = list()
#         tw_attrs, usr_attrs = attr_dict[filter_level]
#         for tw in twarr:
#             if tw_cond(tw):
#                 tw = filter_attribute(tw, tw_attrs)
#                 if tk.key_user in tw:
#                     tw[tk.key_user] = filter_attribute(tw[tk.key_user], usr_attrs)
#                 res_twarr.append(tw)
#         return res_twarr
#
#
# def filter_twarr_text(twarr,
#                       tw_cond=lambda tw: tk.key_orgntext in tw or tk.key_text in tw,
#                       get_text=lambda tw: tw[tk.key_orgntext] if tk.key_orgntext in tw else tw[tk.key_text],
#                       mov_text=lambda tw, text: tw.setdefault(tk.key_orgntext, text),
#                       flt_text=lambda text: pu.text_normalization(text),
#                       set_text=lambda tw, text: tw.setdefault(tk.key_text, text)):
#     for _tw in twarr:
#         if tw_cond(_tw):
#             text = get_text(_tw)
#             mov_text(_tw, text)
#             text = flt_text(text)
#             set_text(_tw, text)
#     return twarr


def filter_twarr_dup_id(twarr, get_id=lambda tw: tw.get(tk.key_id)):
    """ An inplace method """
    dup_idx_list = sorted(twarr_dup_id(twarr, get_id))
    for _idx in range(len(dup_idx_list)-1, -1, -1):
        del twarr[dup_idx_list[_idx]]
    return twarr, dup_idx_list


def filter_attribute(target_dict, attr_set):
    if attr_set is None:
        return target_dict
    for tw_attr in set(target_dict.keys()):
        if tw_attr not in attr_set:
            target_dict.pop(tw_attr)
    return target_dict


def filter_text(text):
    return pu.text_normalization(text)
