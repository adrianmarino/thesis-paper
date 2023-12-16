from langchain.schema import BaseOutputParser
from bunch import Bunch
from typing import List
import util as ut
import re


class MovieRecommenderOutputParser(BaseOutputParser[List[str]]):
    def parse(self, text: str) -> List[str]:
        recs = ut.between(
            text.lower(),
            'comienza lista de recommendaciones:',
            'fin de lista de recommendaciones:'
        )


        results = []
        for line in recs.split('\n'):
            data = re.split(r'\(|\)\:', line, 2)

            if len(data) <= 1:
                continue

            (position, title) = re.split(r'\.', data[0], 1)
            (release, rating) = re.split(r',', data[1], 1)

            results.append({
                'position'   : int(position),
                'title'      : title.strip().capitalize(),
                'release'    : int(release),
                'rating'     : float(rating),
                'description': data[2].strip().capitalize()
            })

        return results