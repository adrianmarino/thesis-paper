import os
import subprocess
import logging
import util as ut

class Mongo:
    @staticmethod
    def command(database, query, host='localhost'):
        query = f'db = connect( "mongodb://{host}/{database}");\n' + query

        commnand_file_path = ut.write('/tmp/query.js', query)

        command = ['mongosh',  '--file', commnand_file_path]

        ut.ProcessHelper.run([command])

    @staticmethod
    def list_collections(database):
        Mongo.command(
            database,
            """
            db.runCommand({
                listCollections: 1.0,
                nameOnly: true
            })
            """
        )

    @staticmethod
    def drop(database, collections):
        for collection in collections:
            Mongo.command(database,  f"db.getCollection('{collection}').drop();")


    @staticmethod 
    def import_csv(database, file_paths):
        commands = []

        for file_path in file_paths:
            commands.append(
                [
                    'mongoimport',
                    '-d',
                    database,
                    '-c',
                    os.path.basename(file_path).split('.csv')[0],
                    '--type',
                    'csv',
                    '--file',
                    file_path,
                    '--headerline'
                ]
            )

        ut.ProcessHelper.run(commands)

        @staticmethod 
    def import_json(database, file_paths):
        commands = []

        for file_path in file_paths:
            commands.append(
                [
                    'mongoimport',
                    '-d',
                    database,
                    '-c',
                    os.path.basename(file_path).split('.json')[0],
                    '--file',
                    file_path
                ]
            )

        ut.ProcessHelper.run(commands)


    @staticmethod
    def export_to_json(database, path, collections):
        commands = []
        for collection in collections:
            commands.append(
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
                ]
            )

        ut.ProcessHelper.run(commands)