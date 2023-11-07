import os
import subprocess
import logging


class ProcessHelper:
    @staticmethod
    def run(commands):
        processes = []

        for command in commands:
            processes.append(
                subprocess.Popen(
                    command, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT
                )
            )

        for p in processes:
            p.wait()
            out, err = p.communicate()
            if err:
                logging.error(err)
            else:
                logging.info('Success: ' + str(out).replace('\\t', '  ').replace('\\n', ''))
