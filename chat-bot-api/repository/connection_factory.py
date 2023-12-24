import os
import motor

class ConnectionFactory:
    @staticmethod
    def create(
        uri=os.environ['MONGODB_URL'], 
        database=os.environ['DATABASE']
    ):
        client = motor.motor_tornado.MotorClient(uri)
        return client.get_database(database)
