import os
import motor

class MongoConnectionFactory:
    @staticmethod
    def create(
        uri=os.environ['MONGODB_URL'],
        database=os.environ['MONGODB_DATABASE']
    ):
        client = motor.motor_tornado.MotorClient(uri)
        return client.get_database(database)
