#!/usr/bin/env python3
"""Pass input directly to output.

https://app.assembla.com/spaces/portaudio/git/source/master/test/patest_wire.c

"""
import argparse
import sounddevice as sd
import numpy  # Make sure NumPy is loaded before it is used in the callback
assert numpy  # avoid "imported but unused" message (W0611)
from modules import (wah_wah, 
                     fuzz, delay, 
                     overdrive, 
                     fuzzexp, 
                     pan, 
                     schroeder1, 
                     compressor, 
                     moorer, 
                     chorus,
                     reverb
                     )

fs = 44100
def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

# the process function!
def process(input_buffer, output_buffer, buffer_len):

    return

def callback(indata, outdata, frames, time, status):
    if status:
        print(status)

    outdata[:] = fuzz(indata, fs) 
    # outdata[:] = delay(indata, fs)
    # outdata[:] = reverb(indata, fs)
    # outdata[:] = schroeder1(indata, fs)
    # outdata[:] = chorus(indata, fs)
    # outdata[:] = indata

def main():
    """
    Input to Output Pass-Through
    """
    
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '-l', '--list-devices', action='store_true',
        help='show list of audio devices and exit')
    args, remaining = parser.parse_known_args()
    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parser])
    parser.add_argument(
        '-i', '--input-device', type=int_or_str,
        help='input device (numeric ID or substring)')
    parser.add_argument(
        '-o', '--output-device', type=int_or_str,
        help='output device (numeric ID or substring)')
    parser.add_argument(
        '-c', '--channels', type=int, default=2,
        help='number of channels')
    parser.add_argument('--dtype', help='audio data type')
    parser.add_argument('--samplerate', type=float, help='sampling rate')
    parser.add_argument('--blocksize', type=int, help='block size')
    parser.add_argument('--latency', type=float, help='latency in seconds')
    args = parser.parse_args(remaining)

    # print(sd.Stream.samplerate)
    try:
        with sd.Stream(device=(args.input_device, args.output_device),
                       samplerate=args.samplerate, blocksize=args.blocksize,
                       dtype=args.dtype, latency=args.latency,
                       channels=args.channels, callback=callback):
            print('#' * 80)
            print('press Return to quit')
            print('#' * 80)
            input()
    except KeyboardInterrupt:
        parser.exit('')
    except Exception as e:
        parser.exit(type(e).__name__ + ': ' + str(e))

if __name__ == '__main__':
    main()