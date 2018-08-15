import argparse

from seeding.seed_parser import *
from seeding.main2parser import *
import utils.function_utils as fu
from config.configure import getcfg


# seed_queries = [
#     # [{'any_of': ['', '', '', '', '', '', '', ],
#     #   'all_of': ['', '', ], }, ['2016', '', ''], ['2016', '', '']],
#
#     # [{'any_of': ['shoot', 'gunman', 'fire', '\Wpub\W', 'attack', 'shot', 'death', 'kill\W'],
#     #   'all_of': ['Aviv', ], }, ['2016', '1', '1'], ['2016', '1', '6']],
#     # [{'any_of': ['knife', 'attack', ],
#     #   'all_of': ['Paris', 'police', 'station', ], }, ['2016', '1', '6'], ['2016', '1', '10']],
#     # [{'any_of': ['explode', 'shopping', 'gunmen', 'car bomb', 'attack', 'injure', ],
#     #   'all_of': ['\WBaghdad\W', ], }, ['2016', '1', '10'], ['2016', '1', '15']],
#     # [{'any_of': ['blast', 'suicide bomb', 'wound', 'attack', 'foreign tourist', ],
#     #   'all_of': ['Istanbul', ], }, ['2016', '1', '11'], ['2016', '1', '16']],
#     # [{'any_of': ['\Wbomb', 'security ', 'suicide', 'polio', 'policemen', ],
#     #   'all_of': ['Quetta', ], }, ['2016', '1', '12'], ['2016', '1', '17']],
#     # [{'any_of': ['siege', 'Union', 'attack', 'massive', ],
#     #   'all_of': ['\WAdde\W', ], }, ['2016', '1', '14'], ['2016', '1', '19']],
#     # [{'any_of': ['offensive', 'ez-Zor', 'militant', 'murder', 'execut', 'soldier', 'civilian', ],
#     #   'all_of': ['Deir', ], }, ['2016', '1', '15'], ['2016', '1', '20']],
#     # [{'any_of': ['gunfire', 'militant', 'restaurant', 'attack', 'civilian', 'Al-Shabaab', 'kill', 'car bomb'],
#     #   'all_of': ['Mogadishu', ], }, ['2016', '1', '21'], ['2016', '1', '26']],
#     # [{'any_of': ['attack', 'car bomb', 'army division', 'soldier', 'tribal fighter', 'offensive', 'kill', ],
#     #   'all_of': ['Ramadi', ], }, ['2016', '1', '26'], ['2016', '1', '31']],
#     # [{'any_of': ['attack', '\Wkill', ],
#     #   'all_of': ['Dalori', 'Boko', ], }, ['2016', '1', '29'], ['2016', '2', '4']],
#     # [{'any_of': ['twin blasts', 'double bomb', 'kill', 'wound', 'Sayeda Zeinab', ],
#     #   'all_of': ['Damascus', ], }, ['2016', '1', '30'], ['2016', '2', '4']],
#     # [{'any_of': ['suicide bomb', 'suicide attack', 'headquarter', 'casualt', 'blast', 'kill', 'injure'],
#     #   'all_of': ['Kabul', ], }, ['2016', '1', '31'], ['2016', '2', '5']],
#     #
#     # [{'any_of': ['Daallo Airline', 'take\s*off', 'emergency land', 'perpetrator', 'injure', ],
#     #   'all_of': ['Somalia', ], }, ['2016', '2', '1'], ['2016', '2', '6']],
#     # [{'any_of': ['machete', 'attack', 'shot\W', 'shoot', 'kill', 'car chase', 'restaurant', 'injure', 'stormed', ],
#     #   'all_of': ['Ohio', ], }, ['2016', '2', '10'], ['2016', '2', '16']],
#     # [{'any_of': ['attack', 'army bus', 'central square', 'military vehicle', 'fire\W', 'explosion', 'parliament'],
#     #   'all_of': ['Ankara', ], }, ['2016', '2', '16'], ['2016', '2', '21']],
#     # [{'any_of': ['\Wbomb', 'blast', '\Wkill', 'suicide', ],
#     #   'all_of': ['Cameroon\W', ], }, ['2016', '2', '18'], ['2016', '2', '23']],
#     # [{'any_of': ['attack', '\Wbomb', '\Wkill', 'suicide', ], 'none_of': ['airstrike', ],
#     #   'all_of': ['Homs', ], }, ['2016', '2', '20'], ['2016', '2', '25']],
#     # [{'any_of': ['explosion', 'suicide bomb', 'SYL hotel', 'Al-Shabaab', 'gunmen', 'kill', 'wound', 'attack', ],
#     #   'all_of': ['Mogadishu', ], }, ['2016', '2', '25'], ['2016', '3', '2']],
#     #
#     # [{'any_of': ['stab', 'shot dead', 'knife', 'attack', 'restaurant', 'Jaffa Port', 'tourist', 'wound', ],
#     #   'all_of': ['Aviv', ], }, ['2016', '3', '7'], ['2016', '3', '12']],
#     # [{'any_of': ['car bomb', 'shooting', 'kill', 'wound', 'gunfire', 'blast explode', 'casualt', '\Wtoll\W', ],
#     #   'all_of': ['Ankara', ], }, ['2016', '3', '12'], ['2016', '3', '17']],
#     # [{'any_of': ['shoot', 'casualt', 'AQIM', '\Wtoll\W', 'gunmen', '\Wkill', 'attack', 'hotel', 'Ivory Coast', 'tourist', 'beach resort', ],
#     #   'all_of': ['Bassam', ], }, ['2016', '3', '12'], ['2016', '3', '18']],
#     # [{'any_of': ['\Wbomb\W', 'bus bomb', 'detonate', '\Wkill', 'injure', 'government employee', ],
#     #   'all_of': ['Peshawar', ], }, ['2016', '3', '15'], ['2016', '3', '20']],
#     # [{'any_of': ['suicide bomb', 'suicide attack', '\Wkill', 'injure', 'explode', 'explosive', 'outskirt', 'Mosque', '\Wbomb'],
#     #   'all_of': ['Maiduguri', ], }, ['2016', '3', '15'], ['2016', '3', '20']],
#     # [{'any_of': ['suicide bomb', 'tourist', '\Wkill', '\Winjure', '\Wblast', '\Wtoll', 'blast', 'explode', ],
#     #   'all_of': ['Istanbul', ], }, ['2016', '3', '18'], ['2016', '3', '23']],
#     # [{'any_of': ['\Wbomb', '\Wkill', 'suicide', ],
#     #   'all_of': ['Brussels', 'attack', ], }, ['2016', '3', '22'], ['2016', '3', '27']],
#     # [{'any_of': ['car bomb', 'suicide bomb', 'explode', 'blast', 'explosion', 'checkpoint', '\Wkill', '\Wwound', ],
#     #   'all_of': ['\WAden\W', ], }, ['2016', '3', '24'], ['2016', '3', '29']],
#     # [{'any_of': ['suicide bomb', 'blast', 'explosion', 'Gulshane Iqbal park', '\Wwound', 'entrance', '\Wtoll'],
#     #   'all_of': ['\WLahore', ], }, ['2016', '3', '26'], ['2016', '3', '31']],
#     # [{'any_of': ['attack', 'Dantewada', ],
#     #   'all_of': ['Chhattisgarh', 'CRPF', ], }, ['2016', '3', '29'], ['2016', '4', '3']],
#     #
#     # [{'any_of': ['\WAbu\W', 'Sayyaf', 'clash', '\Wgun\W', 'militant', ],
#     #   'all_of': ['Philippine', 'soldier', ], }, ['2016', '4', '8'], ['2016', '4', '13']],
#     # [{'any_of': ['explosion', '\Wbomb', '\Wattack', '\Wbus\W', ],
#     #   'all_of': ['Jerusalem', ], }, ['2016', '4', '17'], ['2016', '4', '22']],
#     # [{'any_of': ['Afghanistan', '\Wbomb', '\Wgun\W', 'security', ],
#     #   'all_of': ['Kabul', 'attack', ], }, ['2016', '4', '18'], ['2016', '4', '23']],
#     # [{'any_of': ['murder', 'raid', 'injure', 'suicide', '\Wbomb', 'explosion', '\Wkill', 'wound', 'troop', 'Boko Haram', 'terrorist'],
#     #   'all_of': ['Borno', ], }, ['2016', '4', '23'], ['2016', '4', '29']],
#     # [{'any_of': ['explosion', 'suicide bomb', 'Mosque', 'historic symbol', '\Wwound', 'militant', 'attack', ],
#     #   'all_of': ['Bursa', ], }, ['2016', '4', '26'], ['2016', '5', '2']],
#     # [{'any_of': ['blast', 'suicide bomb', 'car bomb', 'wounded', 'checkpoint', 'bomb attack', ],
#     #   'all_of': ['Baghdad', ], }, ['2016', '4', '29'], ['2016', '5', '4']],
#     # [{'any_of': ['attack', '\Wbomb', 'military', 'recapture', 'Shabaab', ],
#     #   'all_of': ['Somalia', ], }, ['2016', '4', '30'], ['2016', '5', '5']],
#     # [{'any_of': ['Car bomb', 'police hq', 'headquater', 'police station', '\Wkilled', '\Wwounded', '\Wexplosi', 'suicide'],
#     #   'all_of': ['Gaziantep', ], }, ['2016', '4', '30'], ['2016', '5', '5']],
#     #
#     # [{'any_of': ['Car bomb', 'police', 'vehicle', 'targeting', '\Wkilled', '\Wwounded', '\Wexplosi', ],
#     #   'all_of': ['Diyarbakir', ], }, ['2016', '5', '9'], ['2016', '5', '14']],
#     # [{'any_of': ['knife', 'railway', ],
#     #   'all_of': ['Munich', 'attack', ], }, ['2016', '5', '9'], ['2016', '5', '14']],
#     # [{'any_of': ['suicide bomb', 'police', '\Wkill', '\Winjure', '\Wattack', ],
#     #   'all_of': ['Mukalla', ], }, ['2016', '5', '14'], ['2016', '5', '19']],
#     # [{'any_of': ['suicide bomb', 'car bomb', 'shooting', '\Wkill', '\Winjure', '\Wwound', '\Wattack', ],
#     #   'all_of': ['Baghdad', ], }, ['2016', '5', '16'], ['2016', '5', '21']],
#     # [{'any_of': ['suicide', 'car bomb', 'military base', '\Wkill', '\Winjure', '\Wwound', '\Wattack', ],
#     #   'all_of': ['Jableh', ], }, ['2016', '5', '22'], ['2016', '5', '27']],
#     # [{'any_of': ['suicide', '\Wbomb', 'gunmen ', '\Wkill', '\Wwounded', 'Ambassador', ],
#     #   'all_of': ['Mogadishu', ], }, ['2016', '5', '31'], ['2016', '6', '7']],
#     #
#     # [{'any_of': ['\Wbomb', 'police bus', '\Wattack', 'gunmen ', '\Wkill', '\Wwounded', 'explode', 'explosion', ],
#     #   'all_of': ['Istanbul', ], }, ['2016', '6', '6'], ['2016', '6', '11']],
#     # [{'any_of': ['\Wshoot', 'gunmen', '\Wattack', '\Wkill', '\Wwounded', 'Max Brenner ', 'restaurant', 'arrest', ],
#     #   'all_of': ['\WAviv', ], }, ['2016', '6', '7'], ['2016', '6', '12']],
#     # [{'any_of': ['gunman', 'hostage', 'shoot', '\Wshot', ],
#     #   'all_of': ['Orlando', 'nightclub', 'attack', ], }, ['2016', '6', '11'], ['2016', '6', '16']],
#     # [{'any_of': ['\Wpolice', '\Wstab', '\Wattack', '\Wwounded', 'hostage', 'knife', '\Wkill', '\Wterror', ],
#     #   'all_of': ['Magnanville', ], }, ['2016', '6', '12'], ['2016', '6', '18']],
#     # [{'any_of': ['attack', '\Wbomb', 'explode', 'hostage', '\Wkill', 'injured', ],
#     #   'all_of': ['Mogadishu', ], }, ['2016', '6', '24'], ['2016', '6', '29']],
#     #
#     # [{'any_of': ['attack', '\Wbomb', 'headquarter', 'security', '\Wkill'],
#     #   'all_of': ['\WAden\W', 'airport'], }, ['2016', '7', '5'], ['2016', '7', '11']],
#     # [{'any_of': ['Bastille', 'crowd', 'gunfire', ],
#     #   'all_of': ['Nice', 'truck'], }, ['2016', '7', '13'], ['2016', '7', '18']],
#     # [{'any_of': ['\Wkill', '\Wbomb', 'suicide', 'Shabaab', ],
#     #   'all_of': ['Mogadishu', ], }, ['2016', '7', '25'], ['2016', '7', '31']],
#     #
#     # [{'any_of': ['Assam', 'Bodo', '\Wgun\W', 'grenade', '\Wkill', 'shoot', ],
#     #   'all_of': ['Kokrajhar', ], }, ['2016', '8', '4'], ['2016', '8', '9']],
#     # [{'any_of': ['attack', '\Wbomb', 'hospital', '\Wkill', 'suicide', ],
#     #   'all_of': ['Quetta', ], }, ['2016', '8', '7'], ['2016', '8', '10']],
#     # [{'any_of': ['attack', '\Wbomb', 'soldier', ],
#     #   'all_of': ['PKK', 'Turkey', ], }, ['2016', '8', '9'], ['2016', '8', '14']],
#     #
#     # [{'any_of': ['attack', 'suicide', 'Ministry', 'Defence', 'injure', 'explosion', 'death', ],
#     #   'all_of': ['Kabul', ], }, ['2016', '9', '4'], ['2016', '9', '9']],
#     # [{'any_of': ['bomb', 'explosion', 'shopping center', 'blast', 'attack', 'explode', 'car\W', ],
#     #   'all_of': ['Baghdad', ], }, ['2016', '9', '9'], ['2016', '9', '14']],
#     # [{'any_of': ['soldier', 'Insurgent', 'attack', 'Shabaab', ],
#     #   'all_of': ['Somalia', ], }, ['2016', '9', '16'], ['2016', '9', '21']],
#     # [{'any_of': ['attack', 'kill', 'Kashmir', 'soldier', ],
#     #   'all_of': ['\WUri\W', ], }, ['2016', '9', '17'], ['2016', '9', '22']],
#     # [{'any_of': ['rocket', 'kill ', 'Daesh', 'wound', ],
#     #   'all_of': ['Kilis', ], }, ['2016', '9', '21'], ['2016', '9', '26']],
#
#     # [{'any_of': ['shoot', 'gunman', '\Wfire', 'school', 'attack', '\Wshot', 'death', 'kill\W'],
#     #   'all_of': ['Mosul', ], }, ['2016', '11', '1'], ['2016', '11', '6']],
#     # [{'any_of': ['bombing', 'explode', 'explosion', 'attack', 'death', 'car bombing'],
#     #   'all_of': ['Diyarbakir', ], }, ['2016', '11', '3'], ['2016', '11', '8']],
#     # [{'any_of': ['bombing', 'explode', 'explosion', 'attack', 'death', 'airfield', 'suicide bomber', 'kill\W'],
#     #   'all_of': ['Bagram', ], }, ['2016', '11', '11'], ['2016', '11', '16']],
#     # [{'any_of': ['suicide bomb', 'death', 'mosque', '\Wkill\W', '\Winjure'],
#     #   'all_of': ['Kabul', ], }, ['2016', '11', '20'], ['2016', '11', '25']],
#     # [{'any_of': ['suicide bomb', 'car bomb', 'death', '\Wkill', '\Winjure', '\Wwound'],
#     #   'all_of': ['Mogadishu', ], }, ['2016', '11', '25'], ['2016', '11', '30']],
#
#     # [{'any_of': ['suicide bomb', '\Wbomb', 'death', '\Wkill', '\Winjure', '\Wwound'],
#     #   'all_of': ['Madagali', ], }, ['2016', '12', '8'], ['2016', '12', '13']],
#     # [{'any_of': ['suicide bomb', 'car bomb', 'death', '\Wkill', '\Winjure', '\Wwound'],
#     #   'all_of': ['Istanbul', ], }, ['2016', '12', '9'], ['2016', '12', '14']],
#     # [{'any_of': ['cathedral', 'explode', 'explosion', 'death', '\Wkill', '\Winjure', '\Wwound'],
#     #   'all_of': ['Cairo', ], }, ['2016', '12', '10'], ['2016', '12', '15']],
#     # [{'any_of': ['suicide bomb', 'explode', 'explosion', 'death', '\Wkill', '\Winjure', '\Wwound'],
#     #   'all_of': ['\WAden', ], }, ['2016', '12', '17'], ['2016', '12', '22']],
#     [{'any_of': ['death', '\Wkill', '\Winjure', 'shooting'],
#       'all_of': ['\WKarak', ], }, ['2016', '12', '17'], ['2016', '12', '22']],
# ]
# seed_parser = SeedParser(seed_queries, theme='Terrorist', description='Describes event of terrorist attack')

seed_queries = [
    [{'any_of': ['tornado', 'struck', 'cyclone', 'extratropical', 'state of emergency', 'record-breaking',
                 'damage', 'destroy', 'supercell', ],
      'all_of': ['Manzanita', ], }, ['2016', '10', '13'], ['2016', '10', '18']],

    [{'any_of': ['wildfire', 'Great Smoky Mountain', 'damage', 'destroy', 'burn', '\Wacre\W', 'evacuat', ],
      'all_of': ['National Park', ], }, ['2016', '11', '23'], ['2016', '11', '30']],
    [{'any_of': ['\Wfire', 'wildfire', 'damage', 'burned', '\Wburn\W', '\Wacre\W', 'evacuat', ],
      'all_of': ['Hayden', ], }, ['2016', '7', '7'], ['2016', '7', '15']],

    [{'all_of': ['earthquake', 'Ecuador'], 'any_of':['hit', 'magnitude', 'strike', 'toll', 'Esmeraldas'], },
     ['2016', '4', '15'], ['2016', '4', '19']],
    [{'all_of': ['Italy', 'earthquake', ], 'any_of':['hit', 'magnitude', 'strike', 'toll', ], },
     ['2016', '8', '24'], ['2016', '8', '28']],
    [{'all_of': ['Zealand', 'earthquake', ], 'any_of':['hit', 'magnitude', ], },
     ['2016', '11', '14'], ['2016', '11', '18']],
    [{'all_of': ['Tanzania', 'earthquake', ], 'any_of':['hit', 'magnitude', 'strike', ], },
     ['2016', '9', '10'], ['2016', '9', '14']],
    [{'all_of': ['Indonesia', 'earthquake', ], 'any_of':['hit', 'magnitude', 'strike', ], },
     ['2016', '12', '6'], ['2016', '12', '10']],
    [{'all_of': ['India', 'earthquake', ], 'any_of':['hit', 'magnitude', 'strike', 'intensity', ], },
     ['2016', '1', '2'], ['2016', '1', '7']],

    [{'any_of': ['Romania', 'Belgium', 'France', 'Seine\W', 'Germany', 'disaster',
                 '\Wdead\W', 'death', 'missing', 'rainfall', 'rescue', ],
      'all_of': ['flood', ], }, ['2016', '5', '27'], ['2016', '6', '7']],
    [{'any_of': ['catastrophic', '\Wdead\W', 'death', 'FEMA', 'flooding', 'rainfall', 'state of emergency', ],
      'all_of': ['Louisiana', 'flood', ], }, ['2016', '8', '10'], ['2016', '8', '17']],
    [{'any_of': ['disaster', '\Wdead\W', 'death', 'missing', 'rainfall', 'rescue', ],
      'all_of': ['Virginia', 'flood', ], }, ['2016', '6', '22'], ['2016', '6', '28']],
    [{'any_of': ['damage', 'disaster', '\Wdead\W', 'death', 'missing', 'rainfall', 'rescue', ],
      'all_of': ['Maryland', 'flood', ], }, ['2016', '7', '29'], ['2016', '8', '7']],
    [{'any_of': ['damage', 'disaster', '\Wdead\W', 'death', 'kill', 'missing', 'rainfall', 'rescue', ],
      'all_of': ['Pakistan', 'flood', ], }, ['2016', '4', '2'], ['2016', '4', '8']],
    [{'any_of': ['damage', 'disaster', '\Wdead\W', 'death', 'missing', 'rainfall', 'rescue', ],
     'all_of': ['hurricane', 'Florida', ]}, ['2017', '1', '1'], ['2017', '12', '1']],
    [{'any_of': ['damage', 'disaster', '\Wdead\W', 'death', 'missing', 'rainfall', 'rescue', ],
      'all_of': ['hurricane', 'Texas', ]}, ['2017', '1', '1'], ['2017', '12', '12']]
]
# seed_parser = SeedParser([], theme='NaturalDisaster', description='Describes event of natural disaster events')


def main():
    input_base = '/home/nfs/cdong/tw/origin/'
    # output_base = '/home/nfs/cdong/tw/seeding/Terrorist/queried/positive/'
    output_base = '/home/nfs/cdong/tw/seeding/NaturalDisaster/positive'
    import utils.timer_utils as tmu
    tmu.check_time()
    parse_query_list(input_base, output_base, seed_queries, process_num=20)
    tmu.check_time()
    return
    
    # if args.unlb:
    #     args.ner = False
    #     query_func = exec_query_unlabelled
    #     parser = unlb_parser
    # elif args.cntr:
    #     query_func = exec_query_counter
    #     parser = cntr_parser
    # else:
    #     query_func = exec_query
    #     parser = seed_parser
    # for p in [seed_parser, unlb_parser, cntr_parser]:
    #     p.set_base_path(args.seed_path)
    #
    # if args.query:
    #     query_func(args.summary_path, parser)
    # if args.ner:
    #     exec_ner(parser)
    # if args.train:
    #     exec_train_with_outer(seed_parser, unlb_parser, cntr_parser)
    # if args.temp:
    #     temp(cntr_parser)
    # if args.matrix:
    #     construct_feature_matrix(seed_parser, unlb_parser, cntr_parser)
    #
    # if args.pre_test:
    #     exec_pre_test(args.test_data_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Seeding information")
    parser.add_argument('--summary_path', default='',
                        help='Filtered tweets organized in days as file XX_XX_XX_XX.sum under this path.')
    parser.add_argument('--seed_path', default='',
                        help='Path for extracted seed instances according to particular query.')
    
    parser.add_argument('--unlb', action='store_true', default=False,
                        help='If query is performed for unlabeled tweets.')
    parser.add_argument('--cntr', action='store_true', default=False,
                        help='If query is performed for counter tweets.')
    
    # parser.add_argument('--query', action='store_true', default=False,
    #                     help='If query tweets from summarized tw files.')
    # parser.add_argument('--ner', action='store_true', default=False,
    #                     help='If perform ner on queried file.')
    # parser.add_argument('--train', action='store_true', default=False,
    #                     help='If train the model according to the queried tweets, with internal logic.')
    # parser.add_argument('--temp', action='store_true', default=False,
    #                     help='Just a temp function.')
    # parser.add_argument('--matrix', action='store_true', default=False,
    #                     help='To obtain the matrix for both train and test twarr.')
    #
    # parser.add_argument('--test_data_path', default=getcfg().test_data_path,
    #                     help='Path for test data from dzs.')
    # parser.add_argument('--pre_test', action='store_true', default=False,
    #                     help='Just a temp function to preprocess data from dzs.')
    return parser.parse_args()


if __name__ == "__main__":
    main()
