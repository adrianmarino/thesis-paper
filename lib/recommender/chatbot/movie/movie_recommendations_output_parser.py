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

            for idx in range(self.__list_size-1):
                try:
                    line = re.findall(f'\n{idx+1}.\s*(.*?)\n{idx+2}.\s*', text)[0]
                except Exception as e:
                    logging.error(f'Error to parse line between {idx+1} and {idx+2}. {e}')
                    return results

                try:
                    data = re.split(r'\(|\)\:', line)

                    if len(data) <= 1:
                        continue

                    results.append({
                        'title'      : data[0].replace('"', '').replace('\"', '"').replace('\d\.', '').strip().capitalize(),
                        'description': data[2].strip().capitalize(),
                        'release'    : data[1].strip()
                    })
                except Exception as e:
                    logging.error(f'Error to parse line: "{line}"')

            return { 'recommendations': results }