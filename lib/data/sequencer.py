


class Sequencer:
    def __init__(self, start=0):
        self.sequence = start
        self.mapping = {}
    
    def get(self, value):
        if value not in self.mapping:
            self.mapping[value] = self.sequence
            self.sequence +=1
        return self.mapping[value]