import soundfile as sf
import numpy as np 
from modules import wah_wah, fuzz, delay, overdrive, fuzzexp, pan, schroeder1, compressor

if __name__ == '__main__':
  
    # Load input audio
    x, fs = sf.read('acoustic.wav')

    # Wah-Wah 
    # wah_output = wah_wah(x, fs)
    # fuzz_output = fuzz(x, fs)
    # wah_fuzz_output = fuzz(wah_output, fs)
    # delay_output = delay(x, fs)
    compressor_output = compressor(x, fs)
    # overdrive_output = overdrive(x, fs)
    # fuzzexp_output = fuzzexp(x, fs)
    # pan_output = pan(x, fs)
    schroeder1_output = schroeder1(x, fs)

    # Write output wav files
    # sf.write('out_wah.wav', wah_output, samplerate = fs)
    # sf.write('out_fuzz.wav', fuzz_output, samplerate = fs)
    # sf.write('out_wah_fuzz.wav', wah_fuzz_output, samplerate = fs)
    # sf.write('out_delay.wav', delay_output, samplerate = fs)
    # sf.write('out_overdrive.wav', overdrive_output, samplerate = fs)
    # sf.write('pan_output.wav', pan_output, samplerate = fs)
    sf.write('compressor_output.wav', compressor_output, samplerate = fs)
