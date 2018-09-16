import argparse
import json
import os
import random
import subprocess
import twitter

prev_cwd = os.getcwd()
os.chdir(os.path.join(os.path.split(__file__)[0], '..'))

parser = argparse.ArgumentParser()
parser.add_argument('--count', type=int, default=32)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('-i', '--interactive', action='store_true', default=False)
args, other_args = parser.parse_known_args()

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
            'bin/x64/' + ('Debug' if args.debug else 'Release') + '/fractal.exe',
            '-re', re,
            '-im', im,
            '-scale', scale
        ]

        if args.interactive:
            fractal_cmd.extend(['-interactive', 'true'])
        if len(other_args):
            fractal_cmd.extend(other_args)

        fractal = subprocess.Popen(fractal_cmd)
        fractal.communicate()
finally:
    os.chdir(prev_cwd)