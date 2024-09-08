import gzip, json, sys

if len(sys.argv) != 3:
    print('Please provide the correct number of arguments')
    print('Example use:')
    print('    python3 evaluate.py path/to/myresults.txt path/to/validation-with-classifications.txt.gz')

RESULTS = sys.argv[1]
TRUTH = sys.argv[2]

print(f'Evaluating {RESULTS} based on {TRUTH}')


def maybe_gz_open(path):
    if path.endswith('.gz'):
        return gzip.open(path, 'rt')
    return open(path)


results = {}
with maybe_gz_open(RESULTS) as fp:
    for line in fp:
        entry = json.loads(line)
        if len(entry['classifications']) > 5:
            raise Exception(f'"{entry["title"]}" has too many classifications')
        results[entry['title']] = entry['classifications']

total_classifications = 0
correctly_guessed = 0
with maybe_gz_open(TRUTH) as fp:
    for line in fp:
        entry = json.loads(line)
        if entry['title'] not in results:
            raise Exception(f'"{entry["title"]}" missing in {RESULTS}')
        total_classifications += len(entry['classifications'])
        correctly_guessed += sum(c in results[entry['title']] for c in entry['classifications'])

print(f'Evaluation score: {correctly_guessed}/{total_classifications} ≈ {correctly_guessed/total_classifications:.3f}')
