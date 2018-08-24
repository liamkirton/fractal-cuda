import json
import os
import subprocess
import twitter

prev_cwd = os.getcwd()
os.chdir(os.path.join(os.path.split(__file__)[0], '..'))

try:
    api = twitter.Api(**json.loads(open('scripts/keys.json', 'r').read()))
    for tweet in api.GetUserTimeline(screen_name='randommandelbot', count=8):
        coords = tweet.text.split(' ')
        re = coords[0].replace('e+00', '')
        im = coords[2][:-1].replace('e+00', '')
        scale = str(1.0/float(coords[5][:-1])).replace('e+00', '')

        print('>>> Random Mandelbrot', 're:', re, 'im:', im, 'scale:', scale)

        fractal = subprocess.Popen([
            'bin/x64/Release/fractal.exe',
            '-re', re,
            '-im', im,
            '-scale', scale
        ])
        fractal.communicate()
finally:
    os.chdir(prev_cwd)