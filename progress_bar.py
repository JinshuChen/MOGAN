import colorama
import tqdm

TQDM_SMOOTHING = 0

def create_progressbar(iterable,
                       desc="",
                       unit="it",
                       initial=0,
                       offset=0,
                       invert_iterations=False,
                       logging_on_update=False,
                       logging_on_close=True,
                       postfix=False,
                       leave=True):
    reset = colorama.Style.RESET_ALL
    bright = colorama.Style.BRIGHT
    cyan = colorama.Fore.CYAN
    green = colorama.Fore.GREEN

    # ---------------------------------------------------------------
    # Specify progressbar layout:
    #   l_bar, bar, r_bar, n, n_fmt, total, total_fmt, percentage,
    #   rate, rate_fmt, rate_noinv, rate_noinv_fmt, rate_inv,
    #   rate_inv_fmt, elapsed, remaining, desc, postfix.
    # ---------------------------------------------------------------
    bar_format = ""
    bar_format += "%s==>%s%s {desc}%s" % (cyan, reset, bright, reset)  # description
    bar_format += "{percentage:3.0f}%"  # percentage
    bar_format += "|{bar}| "  # bar
    bar_format += " {n_fmt}/{total_fmt}  "  # i/n counter
    bar_format += "{elapsed}<{remaining}"  # eta
    if invert_iterations:
        bar_format += " {rate_inv_fmt}  "  # iteration timings
    else:
        bar_format += " {rate_noinv_fmt}  "
    bar_format += "%s{postfix}%s" % (green, reset)  # postfix

    # ---------------------------------------------------------------
    # Specify TQDM arguments
    # ---------------------------------------------------------------
    tqdm_args = {
        "iterable": iterable,
        "desc": desc,  # Prefix for the progress bar
        "total": len(iterable),  # The number of expected iterations
        "leave": True,  # Leave progress bar when done
        "miniters": 1,  # Minimum display update interval in iterations
        "unit": unit,  # String be used to define the unit of each iteration
        "initial": initial,  # The initial counter value.
        "dynamic_ncols": True,  # Allow window resizes
        "smoothing": TQDM_SMOOTHING,  # Moving average smoothing factor for speed estimates
        "bar_format": bar_format,  # Specify a custom bar string formatting
        "position": offset,  # Specify vertical line offset
        "ascii": True,
        "logging_on_update": logging_on_update,
        "logging_on_close": logging_on_close,
        "leave": leave  # whether to leave the bar when the process finished
    }

    return tqdm_with_logging(**tqdm_args)

# -----------------------------------------------------------------
# Subclass tqdm to achieve two things:
#   1) Output the progress bar into the logbook.
#   2) Remove the comma before {postfix} because it's annoying.
# -----------------------------------------------------------------
class TqdmToLogger(tqdm.tqdm):
    def __init__(self, iterable=None, desc=None, total=None, leave=True,
                 file=None, ncols=None, mininterval=0.1,
                 maxinterval=10.0, miniters=None, ascii=None, disable=False,
                 unit='it', unit_scale=False, dynamic_ncols=False,
                 smoothing=0.3, bar_format=None, initial=0, position=None,
                 postfix=None,
                 logging_on_close=True,
                 logging_on_update=False):

        self._logging_on_close = logging_on_close
        self._logging_on_update = logging_on_update
        self._closed = False

        super(TqdmToLogger, self).__init__(
            iterable=iterable, desc=desc, total=total, leave=leave,
            file=file, ncols=ncols, mininterval=mininterval,
            maxinterval=maxinterval, miniters=miniters, ascii=ascii, disable=disable,
            unit=unit, unit_scale=unit_scale, dynamic_ncols=dynamic_ncols,
            smoothing=smoothing, bar_format=bar_format, initial=initial, position=position,
            postfix=postfix)

    @staticmethod
    def format_meter(n, total, elapsed, ncols=None, prefix='', ascii=False,
                     unit='it', unit_scale=False, rate=None, bar_format=None,
                     postfix=None, unit_divisor=1000, nrows=None):

        meter = tqdm.tqdm.format_meter(
            n=n, total=total, elapsed=elapsed, ncols=ncols, prefix=prefix, ascii=ascii,
            unit=unit, unit_scale=unit_scale, rate=rate, bar_format=bar_format,
            # postfix=postfix, unit_divisor=unit_divisor, nrows=nrows)
            postfix=postfix, unit_divisor=unit_divisor)

        # get rid of that stupid comma before the postfix
        if postfix is not None:
            postfix_with_comma = ", %s" % postfix
            meter = meter.replace(postfix_with_comma, postfix)

        return meter

    def update(self, n=1):
        if self._logging_on_update:
            msg = self.__repr__()
            # logging.logbook(msg)
        return super(TqdmToLogger, self).update(n=n)

    def close(self):
        if self._logging_on_close and not self._closed:
            msg = self.__repr__()
            # logging.logbook(msg)
            self._closed = True
        return super(TqdmToLogger, self).close()


def tqdm_with_logging(iterable=None, desc=None, total=None, leave=True,
                      ncols=None, mininterval=0.1,
                      maxinterval=10.0, miniters=None, ascii=None, disable=False,
                      unit="it", unit_scale=False, dynamic_ncols=False,
                      smoothing=0.3, bar_format=None, initial=0, position=None,
                      postfix=None,
                      logging_on_close=True,
                      logging_on_update=False):
    return TqdmToLogger(
        iterable=iterable, desc=desc, total=total, leave=leave,
        ncols=ncols, mininterval=mininterval,
        maxinterval=maxinterval, miniters=miniters, ascii=ascii, disable=disable,
        unit=unit, unit_scale=unit_scale, dynamic_ncols=dynamic_ncols,
        smoothing=smoothing, bar_format=bar_format, initial=initial, position=position,
        postfix=postfix,
        logging_on_close=logging_on_close,
        logging_on_update=logging_on_update)