import sys
import time
import os

class Logger():
    def __init__(self, path, append, fields, formats=None):
        self.fields = fields
        self.path = path
        self.append = append
        self.file = open(path,'a' if append else 'w' ) 

        if not append:
            self.file.write(','.join(self.fields) + '\n')

        if formats is None:
            self.format_string = None
        else:
            if len(fields) != len(formats):
                raise Exception("Field and format lists must be the same length")
            self.format_string = ','.join(["{{{}:{}}}".format(fld,frmt) for fld,frmt in zip(fields,formats)])

    def log(self, data):
        # Reject with missing fields
        for f in self.fields:
            if not f in data:
                raise Exception("Field not found in supplied data")

        if self.format_string is None:
            self.file.write(','.join([str(data[f]) for f in self.fields]) + '\n')

        else:
            self.file.write(self.format_string.format(**data) + '\n')

    def __del__(self):
        if not self.file.closed:
            self.file.close()
