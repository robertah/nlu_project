import yaml
import os

def main():
    # load variables from config.yml
    with open('config.yaml', 'r') as stream:
        config = yaml.load(stream)

    '''    Testing if it words
    print(config['token']['bos'])

    for file in os.listdir(config['path']['data']):
        if file.endswith(".txt"):
            print(file)
    '''

main()