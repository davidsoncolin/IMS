import IO
import time
import zmq
from zmq import Poller

__author__ = 'DavidJ'

"""
Class to serve and broadcast any type of motion-capture data, provided it is
in a "plain-old-data" format (i.e. lists,dictionaries,floats,strings and integers)
with a dictionary at the top level.
"""


class ReframeServer:
    def __init__(self, getFrame, hashState, hostName, portPub, portRep):
        """
        :zero argument function getFrame:    function to get next data frame - this will be system specific
        :zero argument function hashState:   hash function for data frame to determine if whole frame should be transmitted
        :string hostName:    machine name
        :int port:        TCP-IP port to which data is published (port number for client/server access is one less)
        """
        self.getFrame = getFrame
        self.hashState = hashState
        self.hostName = hostName
        # self.port = port
        self.server = zmq.Context().socket(zmq.PUB)
        self.server.bind('tcp://%s:%d' % (hostName, portPub))  # reframe server makes itself available as "publisher""
        self.command = zmq.Context().socket(zmq.REP)
        self.command.bind('tcp://%s:%d' % (hostName, portRep))  # reframe server makes itself available as "server"
        self.lastTime = time.time()  # time of last publish
        self.state = {}
        self.stateHash = None
        self.started = False

    def start(self):
        self.started = True
        self.main_loop()

    def stop(self):
        self.started = False

    def deleteHash(self, frameHash):
        tmp = self.state.pop(frameHash)  # remove entry
        del tmp  # removed for a second time ?

    def setHash(self, frameHash, frame):
        if frameHash != self.stateHash:
            if not self.state.has_key(frameHash):
                frame['hash'] = frameHash  # set 'hash' in Frame
                self.state[frameHash] = frame  # add frame to state
            self.stateHash = frameHash

    def respond(self):
        print ">> ReframeServer::respond..."
        cmd = self.command.recv()
        print ">> ReframeServer::respond: cmd =", cmd
        self.command.send('greetings')
        # if self.command.poll(timeout=0):  # listen - is a request immediately available ?
        #     cmd = self.command.recv()
        #     print 'cmd', cmd
        #     try:
        #         self.command.send(IO.wrap(eval(cmd)))
        #     except:
        #         self.command.send('fail')

    def publish(self, frame):
        self.server.send(frame)
        self.lastTime = time.time()

    def testTimeout(self):
        return time.time() > self.lastTime + 0.02  # is it 20 mS since the last publish ?

    def animDict(self, frame, hash):
        frame['hash'] = hash
        return frame

    def wrap(self, frame, frameHash):
        return IO.wrap(self.animDict(frame, frameHash))

    def main_loop(self):
        self.current_frame = -1
        all_frames = []

        poller = Poller()

        while self.started:
            socks = dict(poller.poll())
            if self.command in socks:
                self.respond()

            if self.server in socks:
                newFrame = False
                frame = self.getFrame()
                if frame is None: continue
                if type(frame) == dict:
                    frameHash = self.hashState(frame)
                    self.setHash(frameHash, frame)
                    all_frames.append((self.wrap(frame, frameHash), frameHash, time.time()))
                    newFrame = True

                # if newFrame or (self.testTimeout() and len(all_frames) > 0):
                if newFrame or len(all_frames) > 0:
                    self.publish(all_frames[self.current_frame][0])
                    if len(all_frames) > 10000:  # limit memory usage
                        delFrame, delHash, delTime = all_frames.pop(0)
                        if delHash not in zip(*all_frames)[1]:
                            self.deleteHash(delHash)
                    elif self.current_frame != -1:
                        self.current_frame += 1
                    if self.current_frame >= len(all_frames): self.current_frame = -1
                self.respond()

    def setFrame(self, frame):  # for moving backwards/forwards in data stream - up to 10,000 frames backwards
        self.current_frame = frame
