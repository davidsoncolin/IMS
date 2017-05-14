import datetime
import time


def dummy_subject(name,n_bones,n_markers):
    rotation    = (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    translation = (0.0,0.0,0.0)
    s = {
       "name"   :       name,
       "bone_Rs":       n_bones * [ rotation ],
       "bone_Ts":       n_bones * [ translation ],
       "bone_names" :   n_bones * [ "bone_name" ],
       "bone_parents" : n_bones * [ "bone_name" ],
       "marker_names" : n_markers *  [ "marker_name" ],
       "marker_Ts"    : n_markers *  [ translation ]
    }
    return s

def dummy_data(base_name, frame_number, n_subjects):
    nine_tuple  = (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    three_tuple = (0.0,0.0,0.0)
    n_bones = 5
    n_markers = 3
    d = {
        "frame_number": frame_number,
        "num_subjects":n_subjects,
        "subjects": [ dummy_subject( base_name + "_" + str(i), n_bones, n_markers) for i in range(n_subjects)  ]
    }
    return d

"""
class to provide dummy motion capture data to exercise reframe
communication framework.

"""
class dummy_data_provider:
    def __init__(self):
        self.then = datetime.datetime.now()
        self.frame_number = 0
        self.n_subjects   = 3
        self.base_name    = "fred"
    def getFrame(self):
        time.sleep(0.1)
        debug = False
        self.frame_number += 1
        now = datetime.datetime.now()
        elapsed = now - self.then
        if ( debug ):         # check out the elapsed time logic
            print "elapsed=",elapsed
        if ( elapsed > datetime.timedelta(seconds=5)):
            self.then = now
            if (self.n_subjects == 3 ):
                 self.n_subjects = 2
                 self.base_name    = "wilma"
            else:
                 self.n_subjects = 3
                 self.base_name    = "fred"

        if self.frame_number > 10: return None
        return dummy_data( self.base_name, self.frame_number, self.n_subjects)


def main():
    provider = dummy_data_provider()
    for i in range(15):
        data = provider.getFrame()
        print("{1}) data = {0}".format(data,i))
        time.sleep(1)

if __name__ == "__main__":
    main()
