import os
import re
    
split_re = re.compile('([a-z]+)\_([\d]+)')
result_re = re.compile('([^.]+).result')
altres_re = re.compile('([^.]+).altres')

# process all outputs
base_directory = 'output'

# ensure output directory exists
if not os.path.exists(base_directory):
    print('No input directory found')
    exit(1)

param_header = ['x', 'y', 't', 'r', 'trial', 'p']
name_header = []
results = {}
is_altres = False

# walk the base directory
for path, _, files in os.walk(base_directory):
    for file in files:
        # only find '*.result' files
        result_match = result_re.fullmatch(file)
        is_altres = False
        if result_match is None:
            result_match = altres_re.fullmatch(file)
            is_altres = True
        if result_match is None:
            continue
        print('Processing ' + path + os.sep + file)
        # get the assignment of the file
        assignment = result_match.group(1)

        # get the parameters
        path_parts = path.split(os.sep)
        param_list = [0] * len(param_header)
        for part in path_parts:
            # split the part
            split_match = split_re.fullmatch(part)
            if split_match is None:
                continue
            # add the parameter
            type = split_match.group(1)
            value = split_match.group(2)
            if type in param_header:
                param_list[param_header.index(type)] = value
        param = tuple(param_list)

        # get the result
        with open(os.path.join(path, file), 'r') as f:
            for line in f:
                # split the line
                line = line.strip()
                line_parts = line.split(',')
                type = line_parts[0]
                value = line_parts[1]
                name = assignment + '_' + type
                if is_altres:
                    name = 'alt_' + name
                # add the result
                if results.get(param) is None:
                    results[param] = {}
                results[param][name] = value
                # add the header if necessary
                if not name in name_header:
                    name_header.append(name)

name_header.sort()
output_file = 'processed.csv'

# write the results
with open(output_file, 'w') as f:
    # write the header
    f.write(','.join(param_header) + ',' + ','.join(name_header) + '\n')
    # for each key value pair in the results
    for param, result in results.items():
        # write the parameter
        f.write(','.join(str(i) for i in param))
        # write the results
        for name in name_header:
            value = result.get(name)
            if value is None:
                value = ''
            f.write(',' + value)
        f.write('\n')