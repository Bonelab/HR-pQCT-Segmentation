'''
Written by Nathan Neeteson.
A class for creating a csv file with a header and then writing data to it.
Initially created for logging training results, but also turned out to be
useful for recording results from test data and morphological analysis.
'''

import csv
from datetime import datetime


class LogWriter(object):

    def __init__(self, filename, cols, parameters=None, title='TRAINING LOG'):

        self.filename = filename
        self.cols = cols
        self.title = title

        self._write_header(parameters)

    def _write_header(self, parameters):

        with open(self.filename, 'w', newline='') as log_file:
            log_writer = csv.writer(log_file, delimiter=',')

            # write out the header
            if parameters:
                log_writer.writerow(['PARAMETERS'])
                for k, v in parameters.items():
                    log_writer.writerow([k, v])

            if self.title:
                log_writer.writerow([self.title])
            log_writer.writerow(self.cols)

    def log(self, data):

        with open(self.filename, 'a', newline='') as log_file:
            log_writer = csv.writer(log_file, delimiter=',')
            log_writer.writerow([data[c] for c in self.cols])


# this logging class works better with the training script that is
# structured properly and takes command line args
# the other class will be deprecated as soon as all of the new code is verified
class Logger(object):

    def __init__(self, filename, fields, args=None, track_time=True, print_out=True):

        self.filename = filename
        self.track_time = track_time
        self.print_out = print_out

        self.fields = {}
        for f in fields:
            self.fields[f] = 0
        if track_time:
            self.fields['time'] = 0

        self._write_header(args)

    def _write_header(self, args):
        with open(self.filename, 'w', newline='') as log_file:
            log_writer = csv.writer(log_file, delimiter=',')

            # document the args if given
            if args:
                log_writer.writerow(['PARAMETERS'])
                for arg in vars(args):
                    log_writer.writerow([arg, getattr(args, arg)])

            # write out the header row for the log data
            log_writer.writerow(self.fields.keys())

    def set_field_value(self, field, value):
        if field in self.fields.keys():
            self.fields[field] = value
        else:
            raise ValueError(f"Logger got field: {field}, which is not in field's dict")

    def log(self):

        if self.track_time:
            self.fields['time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.filename, 'a', newline='') as log_file:

            log_writer = csv.writer(log_file, delimiter=',')
            log_writer.writerow(list(self.fields.values()))

        if self.print_out:
            print(', '.join([f'{k}: {v}' for k, v in self.fields.items()]))
