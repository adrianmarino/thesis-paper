import json

def to_json(obj, sort_keys=True, indent=4, separators=(',', ':')):
        return json.dumps(obj, default=lambda o: o.__dict__, sort_keys=sort_keys, indent=indent)


def to_minified_json(obj, sort_keys=True, indent=4, separators=(',', ':')):
        return json.dumps(obj, default=lambda o: o.__dict__, sort_keys=sort_keys, indent=indent, separators=separators)