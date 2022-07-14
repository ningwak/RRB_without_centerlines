from collections import namedtuple, defaultdict
import itertools
import json
import random
import pdb
import numpy as np
import pdb


TrackRow = namedtuple('Row', ['frame', 'pedestrian', 'x', 'y'])

class ObsConverter(object):
    def __init__(self):
        self.tracks_by_frame = defaultdict(list)

    def obs_to_track_rows(self, frame, obs):
        for i in range(0, len(obs) ):
            # print(obs[i])
            if obs[i][0] != 0:
                '''if i == 0:
                    i1 = 1
                    row = TrackRow(frame, i1, obs[i][2] + 2, obs[i][1])
                elif i == 1:
                    i1 = 0
                    row = TrackRow(frame, i1, obs[i][2] + 2, obs[i][1])
                else:
                    row = TrackRow(frame, i, obs[i][2] + 2, obs[i][1])'''
                row = TrackRow(frame, i, -obs[i][2], obs[i][1])

                # print('row:', row)
                self.tracks_by_frame[row.frame].append(row)

    @staticmethod
    def track_rows_to_paths(primary_pedestrian, track_rows):
        paths = defaultdict(list)
        for row in track_rows:
            paths[row.pedestrian].append(row)
        # if(track_rows[1].frame == 885):
        #   pdb.set_trace()
        # list of paths with the first path being the path of the primary pedestrian
        primary_path = paths[primary_pedestrian]
        other_paths = [path for ped_id, path in paths.items() if ped_id != primary_pedestrian]
        return [primary_path] + other_paths

    def scene(self, obss, frames):
        i = 0
        for frame in frames:
            self.obs_to_track_rows(frame, obss[i])
            '''for j in range(0, len(obss[i])):
                # print(obs[i])
                if obss[i][j][1] != 0:
                    row = TrackRow(frame + 2, j, (obss[i][j][2] + obss[i + 1][j][2])/2 + 2, (obss[i][j][1] + obss[i + 1][j][1])/2)

                    # print('row:', row)
                    self.tracks_by_frame[row.frame].append(row)'''

            i = i + 1
        track_rows = [r
                      for frame in frames
                      for r in self.tracks_by_frame.get(frame, [])]

        # return as paths
        paths = self.track_rows_to_paths(primary_pedestrian=0, track_rows=track_rows)  # returns the paths(frames from first to end of a specific pedestrian) of different pedestrians, the first one is the path of ped on interest and then other ones.(so it is [[trajnettools.data.Rows of ped interest],[trajnettools.data.Row of next ped], ...]]
        '''if (paths[0][1].frame - paths[0][0].frame == 0):
            pdb.set_trace()'''
        return paths
