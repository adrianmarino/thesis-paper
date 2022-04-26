import os
import subprocess
import logging


class Mongo:
    @staticmethod
    def import_csv(database, file_paths):
        processes = []
        for file_path in file_paths:
            collection = os.path.basename(file_path).split('.csv')[0]

            processes.append(subprocess.Popen(
                [
                    'mongoimport', 
                    '-d', 
                    database,
                    '-c',
                    collection,
                    '--type',
                    'csv',
                    '--file',
                    file_path,
                    '--headerline'
                ], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT
            ))

        for p in processes:
            p.wait()
            out, err = p.communicate()
            if err:
                logging.error(err)
            else:
                logging.error(out)

    @staticmethod
    def export_to_json(database, path, collections):
        processes = []
        for collection in collections:
            processes.append(subprocess.Popen(
                [
                    'mongoexport', 
                    '-d', 
                    database,
                    '-c',
                    collection,
                    '--jsonArray',
                    '--pretty',
                    '--out',
                    f'{path}/{collection}.json'
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            ))

        for p in processes:
            p.wait()
            out, err = p.communicate()
            if err:
                logging.error(err)
            else:
                logging.error(out)



