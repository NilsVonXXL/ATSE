import re

input_folder = 'input-x-0.1-y-0.1-eps-0.35'
match = re.match(r'input-x-([-\d.]+)-y-([-\d.]+)-eps-([-\d.]+)', input_folder)
if match:
    x = float(match.group(1))
    y = float(match.group(2))
    eps = float(match.group(3))
    inputs = [x, y, eps]
    print('Inputs:', inputs)
    print('Inputs length:', len(inputs))
else:
    print('Could not parse input folder name!')