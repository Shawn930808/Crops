import sys

class ColourPrinter():
    def __init__(self, stream=sys.stdout):
        self.stream = stream;
        self.use_colour = self._has_colours(self.stream)
        self.BLACK, self.RED, self.GREEN, self.YELLOW, \
        self.BLUE, self.MAGENTA, self.CYAN, self.WHITE = range(30,38)
        
    def _has_colours(self, stream):
        if not hasattr(stream, "isatty"):
            return False
        if not stream.isatty():
            return False # auto color only on TTYs
        try:
            import curses
            curses.setupterm()
            return curses.tigetnum("colors") > 2
        except:
            # guess false in case of error
            return False

    def write(self, text, colour=None):
        if colour is None or not self.use_colour:
            self.stream.write(text)
        else:
            seq = "\x1b[1;%dm" % (colour) + text + "\x1b[0m"
            self.stream.write(seq)

                




