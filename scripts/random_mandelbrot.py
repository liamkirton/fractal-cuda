import json
import subprocess
import twitter

keys = json.loads(open('keys.json', 'r').read())

api = twitter.Api(**keys)

for tweet in api.GetUserTimeline(screen_name='randommandelbot', count=8):
    coords = tweet.text.split(' ')
    re = coords[0]
    im = coords[2][:-1]
    scale = str(1.0/float(coords[5][:-1]))

    print(re, im, scale)

    for cm in ['2', '3', '4', '5', '6', '7']:
        fractal = subprocess.Popen([
            '../bin/x64/Release/fractal.exe',
            '-re', re,
            '-im', im,
            '-scale', scale,
            '-4k', '-el', '262144', '-cm', cm
        ], cwd='..')
        fractal.communicate()