"""Preserved integrated-figure export policy, independent of GUI frameworks."""


def save_molecule_figure(figure, path):
    """Use the original renderer/backend for PNG, PDF and SVG output.

    Do not replace vector formats with a GPU screenshot or alter the figure view.
    The caller owns the dialog, document affinity and error reporting.
    """
    figure.savefig(path, dpi=600, bbox_inches="tight")
