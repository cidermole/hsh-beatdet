
# Hints:
# Detrending uses heuristic beat detection, too.
# Removal of ringing due to low blood pressure may be necessary before actual detection.
# Design the preprocessing steps in a way that each can be debugged separately.


class Detector(object):
    """
    Unified beat detector interface.
    Implementations should include all preprocessing steps, so it can be applied as a uniform interface
    across different sampling rates, signal sources (MIMIC vs. Android camera vs. iOS camera).
    """
    def __init__(self):
        """params: eventually the hyperparams for specific detectors."""
        pass

    def detect(self, ppg_raw):
        """
        Run the beatdetection algorithm.
        Results in quality estimates, noisy sections, good beat locations, etc.
        :param ppg_raw: evenly sampled photoplethysmography signal - camera brightness of red channel.
        """
        raise NotImplementedError()

    def plot(self):
        """Show the debug plot."""
        raise NotImplementedError()

    def get_result(self):
        """Get the actual signal with beats."""
        raise NotImplementedError()
