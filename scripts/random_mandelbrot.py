import argparse
import json
import os
import random
import subprocess
import twitter

prev_cwd = os.getcwd()
os.chdir(os.path.join(os.path.split(__file__)[0], '..'))

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--count', type=int, default=16)
parser.add_argument('-i', '--interactive', action='store_true', default=False)
parser.add_argument('--4k', action='store_true', default=False)
args = parser.parse_args()

try:
    api = twitter.Api(**json.loads(open('scripts/keys.json', 'r').read()))
    tweet_list = list(api.GetUserTimeline(screen_name='randommandelbot', count=args.count))

    if args.interactive:
        tweet_list = [random.choice(tweet_list)]

    for tweet in tweet_list:
        coords = tweet.text.split(' ')
        re = coords[0].replace('e+00', '')
        im = coords[2][:-1].replace('e+00', '')
        scale = str(1.0/float(coords[5][:-1])).replace('e+00', '')

        print('>>> Random Mandelbrot', 're:', re, 'im:', im, 'scale:', scale)

        fractal_cmd =[
            'bin/x64/Release/fractal.exe',
            '-re', re,
            '-im', im,
            '-scale', scale
        ]

        if getattr(args, '4k'):
            fractal_cmd.extend(['-image_width', '3840', '-image_height', '2160'])
        if args.interactive:
            fractal_cmd.extend(['-interactive', 'true'])

        fractal = subprocess.Popen(fractal_cmd)
        fractal.communicate()
finally:
    os.chdir(prev_cwd)