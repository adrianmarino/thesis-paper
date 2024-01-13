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

            for idx in range(1, self.__list_size+1):
                try:
                    line = re.findall(f'\n{idx}.\s*(.*?)\n', text)[0]
                except Exception as e:
                    try:
                        line = re.findall(f'\n\n{idx}.(.*?)\n', text)[0]
                    except Exception as e:
                        try:
                            line = re.findall(f'\n{idx}.(.*?)\.', text)[0]
                        except Exception as e:
                            try:
                                line = re.findall(f'\n\n{idx}.(.*?)\.', text)[0]
                            except Exception as e:
                                logging.error(f'Error to parse line {idx}. {e}')
                                # logging.error(f'Text: {text}')
                                return { 'recommendations': results }


                try:
                    data = re.split(r'\(|\)\:', line)

                    if len(data) <= 1:
                        continue

                    results.append(self._build_item(idx, data))
                except Exception as e:
                    try:
                        data = re.split(r'\(|\)\s*-', line)

                        if len(data) <= 1:
                            continue

                        results.append(self._build_item(idx, data))
                    except Exception as e:
                        logging.error(f'Error to parse line: "{line}"')


            return { 'recommendations': results }

    def _build_item(self, idx, data):
        return {
            'number'     : idx,
            'title'      : data[0].replace('"', '').replace('\"', '"').replace('\d\.', '').strip().capitalize(),
            'description': data[2].strip().capitalize(),
            'release'    : data[1].strip()
        }