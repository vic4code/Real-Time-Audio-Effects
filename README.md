# Real-Time-Audio-Effects
Here is a Python implementation of common audio effects that simulate guitar pedals. Most of the code is refered from [Digital Audio Effects by Marshall et. al](https://www.google.com/searchq=cardiff+university+digitall+audio+effect&oq=cardiff&aqs=chrome.2.69i57j46i67i275j69i59l2j35i39j69i60l3.3778j0j7&sourceid=chrome&ie=UTF-8) which implements the algorithms in Matlab.

The gui is bulid with [Figma](https://www.figma.com/) and transfered in tkinter app with [TKinterDesigner](https://github.com/ParthJadhav/Tkinter-Designer).

<img width="892" alt="Screen Shot 2021-10-17 at 2 15 46 AM" src="https://user-images.githubusercontent.com/45786393/137597986-fb6f92b0-e2d6-47c9-b8b9-59ca83b25452.png">

## Installation

* Clone the repo:
```
git clone https://github.com/vic4code/Real-Time-Audio-Effects.git
```
* Create the environment with Conda:
```
conda env create -f environment.yml
conda activate rt-audio-effect
```

If you would like to do some basic testing, batch and realtime processing scripts are both provided in :
* `batch_process.py`
* `realtime_process.py`

If you want to see the final result, please directly run `main.py` to launch the app:
```
python main.py
```

## Disscusion
When the input sound passes through spatial effects such as "Wah-Wah", "Reverb", "Delay", "Chorus"... There are some glitches when the input audio is passed through, and I think the problem is caused by the buffer IO in the real time processing. The length of the buffer (e.g. 512 samples) is too short to use [Digital Audio Effects](https://www.google.com/searchq=cardiff+university+digitall+audio+effect&oq=cardiff&aqs=chrome.2.69i57j46i67i275j69i59l2j35i39j69i60l3.3778j0j7&sourceid=chrome&ie=UTF-8) in the algorithm for good processing. I think I need to use some post-processing methods to solve this problem.
* Modify the algorithm to consider multiple buffers to accommodate real-time processing.
* Create a fixed queue and process audio by queue instead of by buffer, and the queue contains overlapping buffers.




