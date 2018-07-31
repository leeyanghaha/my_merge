import time


default_name = 'main'
check_points = dict()


def check_time(name=default_name, print_func=lambda dt: print('{} s from last check'.format(dt))):
    if name not in check_points:
        check_points[name] = time.time()
        return None
    else:
        delta_t = time.time() - check_points[name]
        if print_func is not None:
            print_func(delta_t)
        check_points[name] = time.time()
        return delta_t


class Alarm:
    def __init__(self):
        self.timestamp = None
        self.initialize_timestamp()
    
    def initialize_timestamp(self):
        self.timestamp = time.time()
    
    def is_time_elapse_bigger_than(self, interval):
        current = time.time()
        diff = current - self.timestamp
        # print('diff={}'.format(diff))
        if not diff > interval:
            return False
        else:
            return True
