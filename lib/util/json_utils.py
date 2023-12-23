import json

def to_json(obj, sort_keys=True, indent=4):
        return json.dumps(obj, default=lambda o: o.__dict__, sort_keys=sort_keys, indent=indent)
