from typing import List
import re
import logging

class MovieRecommendationsOutputParser():
    def parse(self, text: str, lines_size: int) -> List[str]:
            results = []

            for idx in range(1, lines_size+1):
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
                                logging.error(f'Error to parse line {idx}. {e}. Text: {text}')
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
                        logging.error(f'Error to parse line: "{line}". Detail: {e}')


            return { 'recommendations': results }

    def _build_item(self, idx, data):
        return {
            'number'     : idx,
            'title'      : data[0] \
                .replace('"', '') \
                .replace('\"', '"') \
                .replace('\d\.', '') \
                .replace('**', '') \
                .strip() \
                .capitalize(),
            'description': data[2].strip().capitalize(),
            'release'    : data[1].strip()
        }