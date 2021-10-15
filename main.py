import soundfile as sf
import numpy as np 
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
import os 

if __name__ == '__main__':
  
    # Load input audio
    x, fs = sf.read('input_files/acoustic.wav')

    print(fs)
    # Wah-Wah 
    # wah_output = wah_wah(x, fs)
    # fuzz_output = fuzz(x, fs)
    # wah_fuzz_output = fuzz(wah_output, fs)
    delay_output = delay(x, fs)
    # compressor_output = compressor(x, fs)
    # overdrive_output = overdrive(x, fs)
    # fuzzexp_output = fuzzexp(x, fs)
    # pan_output = pan(x, fs)
    # schroeder1_output = schroeder1(x, fs)
    # moorer_output = moorer(x, fs)
    # chorus_output = chorus(x, fs)
    reverb_output = reverb(x, fs)

    # Write output wav files
    outdir = "output_files"
    # sf.write(os.path.join(outdir, 'wah_output.wav'), wah_output, samplerate = fs)
    # sf.write('out_wah_fuzz.wav', wah_fuzz_output, samplerate = fs)
    # sf.write(os.path.join(outdir, 'out_delay.wav'), delay_output, samplerate = fs)
    # sf.write('out_overdrive.wav', overdrive_output, samplerate = fs)
    # sf.write('pan_output.wav', pan_output, samplerate = fs)
    # sf.write(os.path.join(outdir, 'schroeder1_output.wav'), schroeder1_output, samplerate = fs)
    # sf.write('output_files/moorer_output.wav', moorer_output, samplerate = fs)
    # sf.write(os.path.join(outdir, 'chorus_output.wav'), chorus_output, samplerate = fs)
    sf.write(os.path.join(outdir, 'reverb_output.wav'), reverb_output, samplerate = fs)
