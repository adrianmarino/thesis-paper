from langchain.schema import BaseOutputParser
from bunch import Bunch
from typing import List
import util as ut
import re
import logging
from pydantic import BaseModel, PrivateAttr


class MovieRecommendationsOutputParser(BaseOutputParser[List[str]]):
    __list_size: int = PrivateAttr(True)

    def __init__(self, list_size):
        super().__init__()
        self.__list_size = list_size

    def parse(self, text: str) -> List[str]:
            results = []
            for idx in range(self.__list_size):
                try:
                    line = ut.between(text, f'\n{idx+1}.', f'\n{idx+2}.')

                except Exception as e:
                    logging.error(f'Error to parse response. {e}')
                    return results

                data = re.split(r'\(|\)\:', line)

                if len(data) <= 1:
                    continue

                (release, rating) = re.split(r',', data[1], 1)

                results.append({
                    'title'      : data[0].strip().replace('"', '').capitalize(),
                    'rating'     : float(rating),
                    'description': data[2].strip().capitalize(),
                    'release'    : int(release)
                })

            return {
                'recommendations': sorted(results, key=lambda it: it['rating'], reverse=True)
            }