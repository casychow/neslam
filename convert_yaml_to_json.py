import yaml
import json

with open('TUM1.yml', 'r') as file:
    configuration = yaml.safe_load(file)

with open('transforms.json', 'w') as json_file:
    json.dump(configuration, json_file)

output = json.dumps(json.load(open('config.json')), indent=2)
print(output)