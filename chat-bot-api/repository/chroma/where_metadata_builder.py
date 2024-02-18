
class WhereMetadataBuilder:

    def __init__(self):
        self.conditions = {}

    def is_in(self, field, values, negate=False):
        if field and values and len(values):
            operator = '$nin' if negate else '$in'
            self.conditions[field] = { f'{operator}': [str(v) for v in values] }
        return self

    def gte(self, field, value):
        if field and value and value > 0:
            self.conditions[field] = { '$gte': value }
        return self

    def build(self):
        if len(self.conditions.keys()) >= 2:
            return { '$and': [ {field: condition }for field, condition in self.conditions.items()] }
        else:
            return self.conditions