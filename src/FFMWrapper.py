# Python 2.7
# Brian Hardenstein
# FFMWrapper class
#
# Makes it possible to use ffmpeg to write directly to videos using python

import os
import signal
import matplotlib.pyplot as plt
import numpy as np

# for writing directly to video
import subprocess

class FFMWrapper(object):

    def __init__(self, _filename, _vid_fps=24, _width=1920, _height=1080):
        '''
        Create class object
        '''

        self.width = _width
        self.height = _height

        self.frame_cache = []

        # a more generalized wrapper would let all of this other stuff be
        # specified too but not needed for mvp
        self.cmdstring = ('ffmpeg',
            '-y', '-r', str(_vid_fps), # overwrite, 30fps
            '-s', '{:d}x{:d}'.format(self.width, self.height), # size of image string
            '-pix_fmt', 'argb', # format
            '-f', 'rawvideo',  '-i', '-', # tell ffmpeg to expect raw video from the pipe
            '-vcodec', 'libx264', '-crf', '25', _filename) # output encoding

        # establish subprocess connection
        self.process = subprocess.Popen(self.cmdstring, stdin=subprocess.PIPE)

    def add_to_cache(self, _buffer):
        '''
        Instead of writing after every single frame is created, add to a cache
        and write out the cache every 30 frames (or whatever the fps is set to)
        '''
        self.frame_cache.append(_buffer)

        if len(self.frame_cache) > 30:
            self.write_cache()

    def write_cache(self):
        '''
        Append the frames in the frame cache to the video
        '''
        for idx in range(len(self.frame_cache)):
            temp_buffer = self.frame_cache.pop(0)

            self.write_frame(temp_buffer)

    def write_frame(self, _buffer):
        '''
        Appends a frame to the video. The frame MUST be an ARGB string!
        Example: _exit_buffer = fig.canvas.tostring_argb()
        '''
        # write to pipe
        self.process.stdin.write(_buffer)

    def add_frame(self, _buffer, desc=""):
        '''
        Add the provided _buffer to the video with the provided desc
        '''
        # make the plot for the frame
        fig, ax = plt.subplots(figsize=(self.width/100.0, self.height/100.0), dpi=100)
        fig.subplots_adjust(0, 0, 1, 1)

        # fix any alpha weirdness by clipping final_frame then setting alpha
        # to max
        final_frame = np.clip(_buffer, 0, 255)

        # draw frame on canvas
        ax.clear()
        ax.set_facecolor("#000000")
        ax.set_xlim(-110.0, -30.0)
        ax.set_ylim(10, 50.0)
        ax.annotate(desc, xy=(-109, 48), size=40, color='#AAAAAA')

        ax.imshow(final_frame.astype(np.uint8), aspect="auto", alpha=1.0, extent=[-110, -30, 10, 50])
        # fig.savefig("../imgs/test/more_heat/{}".format("test.png"), pad_inches=0, transparent=True)
        fig.canvas.draw()

        # write the frame:
        self.add_to_cache(fig.canvas.tostring_argb())

    def cross_fade_frames(self, start_buffer, end_buffer, num_frames, desc, desc_size=40):
        '''
        Fade the two buffers into each other.
        '''
        # make the plot for the frame
        fig, ax = plt.subplots(figsize=(self.width/100.0, self.height/100.0), dpi=100)
        fig.subplots_adjust(0, 0, 1, 1)

        for idx in range(num_frames):


            final_frame = np.zeros((start_buffer.shape), dtype=float)

            # make temp variables to keep the buffers from getting messed up
            temp_start = start_buffer.copy()
            temp_end = end_buffer.copy()

            # add the buffers to the final_frame buffer
            # percent of frame visible:
            alpha = idx / (num_frames * 1.0)

            final_frame = temp_start * (1.0 - alpha)
            final_frame += temp_end * alpha

            # print "final_frame:", final_frame[1,:]
            # print "final frame shape:", final_frame.shape

            # del temp_start, temp_end

            # fix any alpha weirdness by clipping final_frame then setting alpha
            # to max
            final_frame = np.clip(final_frame, 0, 255)

            # set all alpha values to 255 (fully opaque)
            final_frame[...,3] = 255

            # draw frame on canvas
            ax.clear()
            ax.set_facecolor("#000000")
            ax.set_xlim(-110.0, -30.0)
            ax.set_ylim(10, 50.0)
            ax.annotate(desc, xy=(-109, 47.5), size=desc_size, color='#AAAAAA')

            ax.imshow(final_frame.astype(np.uint8), aspect="auto", alpha=1.0, extent=[-110, -30, 10, 50])
            # fig.savefig("../imgs/test/more_heat/{}".format("test.png"), pad_inches=0, transparent=True)
            fig.canvas.draw()

            # write the frame:
            self.add_to_cache(fig.canvas.tostring_argb())

        plt.close("all")


        # del start_buffer, end_buffer


    def close(self):
        '''
        Close out the write to the process
        '''
        # make sure the frame cache is empty
        if len(self.frame_cache) != 0:
            self.write_cache()

        # try to write the video
        self.process.communicate()

        self.process.kill()
