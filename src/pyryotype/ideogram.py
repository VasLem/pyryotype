"""Ideogram plotting executables."""
from enum import Enum
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.typing import ColorType

from pyryotype.plotting_utils import set_xmargin
import re
from typeguard import check_type
import numpy as np
from matplotlib.gridspec import GridSpec as gs


class GENOME(Enum):
    HG19 = "hg19"
    HG38 = "hg38"
    CHM13 = "chm13"
    HS1 = "hs1"


class Orientation(Enum):
    VERTICAL = "Vertical"
    HORIZONTAL = "Horizontal"


class Detail(Enum):
    CYTOBAND = "Cytoband"
    BARE = "Bare"


COLOUR_LOOKUP = {
    "gneg": (1.0, 1.0, 1.0),
    "gpos25": (0.6, 0.6, 0.6),
    "gpos50": (0.4, 0.4, 0.4),
    "gpos75": (0.2, 0.2, 0.2),
    "gpos100": (0.0, 0.0, 0.0),
    # 'acen': (.8, .4, .4),
    # Set acen to be white as we use a
    #   polygon to add it in later
    "acen": (1.0, 1.0, 1.0),
    "gvar": (0.8, 0.8, 0.8),
    "stalk": (0.9, 0.9, 0.9),
}
STATIC_PATH = Path(__file__).parent / "static"


def get_cytobands(genome: GENOME) -> Path:
    """
    Return the cytobands file for the given genome.

    :param genome: The genome variant to get the cytobands file for.
    :return: The path to the cytobands file associated with the provided genome variant.
    :raises ValueError: If the provided genome variant is not recognized.

    >>> get_cytobands(GENOME.HG38) # doctest: +ELLIPSIS
    PosixPath('/.../src/pyryotype/static/cytobands_hg38.bed')
    >>> get_cytobands(GENOME.CHM13) # doctest: +ELLIPSIS
    PosixPath('/.../src/pyryotype/static/cytobands_chm13.bed')
    >>> get_cytobands(GENOME.HS1) # doctest: +ELLIPSIS
    PosixPath('/.../src/pyryotype/static/cytobands_chm13.bed')
    >>> get_cytobands("invalid_genome")
    Traceback (most recent call last):
    ...
    ValueError: Unknown genome: invalid_genome
    """
    match genome:
        case GENOME.HG19:
            return STATIC_PATH / "cytobands_hg19.bed"
        case GENOME.HG38:
            return STATIC_PATH / "cytobands_hg38.bed"
        case GENOME.CHM13:
            return STATIC_PATH / "cytobands_chm13.bed"
        case GENOME.HS1:
            return STATIC_PATH / "cytobands_chm13.bed"
        case _:
            msg = f"Unknown genome: {genome}"
            raise ValueError(msg)


CHR_PATT = re.compile(r"^(?:chr([0-9]+|x|y|m)(.*)|(.*))$")

def chr_to_ord(x: str):
    out = re.match(CHR_PATT, x.lower())
    if out[3]:
        return (out[3], 0, "")
    chr = out[1].lower()
    if chr == "x":
        chr = 23
    elif chr == "y":
        chr = 24
    elif chr == "m":
        chr = 25
    if out[2]:
        return ("chr", int(chr), out[2])
    return (".", int(chr), "")  # make sure the canonical part stays on top

def get_cytoband_df(genome: GENOME, relative: bool) -> pd.DataFrame:
    """
    Convert the cytogram file for the given genome into a dataframe.
    :param genome: The genome to plot the ideogram for.
    :return: A DataFrame containing chromosome cytoband details.

    >>> dummy_genome = GENOME.HG38  # replace with a test genome path or identifier
    >>> result_df = get_cytoband_df(dummy_genome)
    >>> result_df["chrom"].tolist()[:2]
    ['chr1', 'chr1']
    >>> result_df["chromStart"].tolist()[:2]
    [0, 2300000]
    >>> result_df["arm"].tolist()[:2]
    ['p', 'p']
    """
    cytobands = pd.read_csv(
        get_cytobands(genome), sep="\t", names=["chrom", "chromStart", "chromEnd", "name", "gieStain"]
    )
    cytobands["arm"] = cytobands["name"].str[0]
    cytobands["colour"] = cytobands["gieStain"].map(COLOUR_LOOKUP)
    cytobands["width"] = cytobands["chromEnd"] - cytobands["chromStart"]
    if relative:
        return cytobands
    # Sort the chromosomes in a canonical order
    cytobands = cytobands.sort_values(
        by="chrom",
        key=lambda x: x.map(chr_to_ord),
    )
    # Add a column for the cumulated sum of the chromosome length..
    cumsumlengths = (cytobands.groupby("chrom",sort=False)["chromEnd"].max() - cytobands.groupby("chrom",sort=False)["chromStart"].min()).cumsum()
    # .. to produce the absolute offset of each chromosome
    offset = pd.Series(np.hstack([[0], cumsumlengths[:-1]]), cumsumlengths.index, name="offset")
    cytobands = cytobands.merge(
        offset,
        left_on="chrom",
        right_index=True,
    )
    cytobands["chromStart"] += cytobands["offset"]
    cytobands["chromEnd"] += cytobands["offset"]
    cytobands = cytobands.drop(columns=["offset"])
    return cytobands




def plot_ideogram(
    ax: Axes,
    target: str,
    genome: GENOME = GENOME.HG38,
    start: int | None = None,
    stop: int | None = None,
    zoom: bool = False,
    lower_anchor: int = 0,
    height: int = 1,
    curve: float = 0.05,
    y_margin: float = 0.05,
    right_margin: float = 0.005,
    left_margin: float = 0.25,
    target_region_extent: float = 0.3,
    y_label: str | None = None,
    vertical: Orientation = Orientation.HORIZONTAL,
    regions: list[tuple[int, int, ColorType]] | None = None,
    cytobands: Detail = Detail.CYTOBAND,
    relative: bool = True,
    adjust_margins: bool = True,
    lims_on_curve: bool = True,
    **kwargs,    
):
    """
    Plot a chromosome ideogram with cytobands and optionally highlight a specific region.

    :param ax: Matplotlib axis object where the ideogram will be plotted.
    :param cytobands_df: DataFrame containing cytoband data with columns "chrom", "chromStart",
      "chromEnd", "gieStain", and "colour".
    :param target: Target chromosome to filter and plot.
    :param start: Starting base pair position for the region of interest (optional).
    :param stop: Ending base pair position for the region of interest (optional).
    :param zoom: Whether to zoom in on the region of interest. If not, only a box will be drawn around that region (default: False).
    :param lower_anchor: Lower anchor point for the ideogram, for outline.
    :param height: Height of the ideogram.
    :param curve: Curve factor for the ideogram edges.
    :param y_margin: Margin for the y-axis.
    :param right_margin: Margin for the right side of the x-axis.
    :param left_margin: Margin for the left side of the x-axis.
    :param target_region_extent: Extent of the target region highlight.
    :param vertical: Orientation of ideogram. False draws horizontal ideograms.
    :param regions: List of regions to colour in on the karyotype. Respects vertical kwarg - a region should
    be a tuple of format (start, stop, colour)
    :param cytobands: Whether to render cytobands
    :param relative: Whether to plot the ideogram in relative coordinates (start of chromosome is 0) (default: True).
    :param adjust_margins: Whether to adjust the margins of the plot (default: True).
    :param lims_on_curve: Whether to set the x-axis limits on the end of the visual (including curve), or the true ends of the chromosome (default: True).

    :return: Updated axis object with the plotted ideogram.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax = plot_ideogram(ax, "chr1", start=50, stop=250, y_label="Chromosome 1")
    >>> ax.get_xlim()  # To test if the ideogram was plotted (not a direct measure but gives an idea)
    (-71574971.325, 256487353.7655)

    # Test behaviour with a non-existent chromosome
    >>> ax = plot_ideogram(ax, "chr_1", start=50, stop=250, y_label="Chromosome 1")# doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: Chromosome chr_1 not found in cytoband data. Should be one of ...

    """
    # TODO: various kwds params for passing through to other methods
    
    # some checks for input before we start
    if start is not None:
        assert stop is not None, "If start is provided, stop must also be provided"
    if stop is not None:
        assert start is not None, "If stop is provided, start must also be provided"
    if start is not None:        
        assert start < stop, "Start must be less than stop"
    if regions is not None:
        assert isinstance(regions, list), "Regions must be a list of tuples"
        for region in regions:
            assert len(region) == 3, "Each region must be a tuple of (start, stop, colour)"
            assert isinstance(region[0], int), "Start must be an integer"
            assert isinstance(region[1], int), "Stop must be an integer"
            assert region[0] < region[1], "Start must be less than stop"
            assert check_type(region[2], ColorType), "Third element must be a colour"
    
    df = get_cytoband_df(genome, relative=relative)
    chr_names = df["chrom"].unique()
    df = df[df["chrom"].eq(target)]
    if df.empty:
        msg = f"Chromosome {target} not found in cytoband data. Should be one of {chr_names}"
        raise ValueError(msg)

    # Beginning with plotting
    yrange = (lower_anchor, height)
    xrange = df[["chromStart", "width"]].values
    chr_start = df["chromStart"].min()
    chr_end = df["chromEnd"].max()
    chr_len = chr_end - chr_start
    ymid = (max(yrange) - min(yrange)) / 2
    if cytobands == Detail.CYTOBAND:
        if vertical == Orientation.VERTICAL:
            yranges = df[["chromStart", "width"]].values
            x_range = (lower_anchor, height)
            face_colours = iter(df["colour"])
            for yrange in yranges:
                ax.broken_barh([(lower_anchor, height - 0.01)], yrange, facecolors=next(face_colours), zorder=1)

            (max(x_range) - min(x_range)) / 2

        else:
            ax.broken_barh(xrange, yrange, facecolors=df["colour"], alpha=0.6)

    # Define and draw the centromere using the rows marked as 'cen' in the 'gieStain' column
    cen_df = df[df["gieStain"].str.contains("cen")]
    cen_start = cen_df["chromStart"].min()
    cen_end = cen_df["chromEnd"].max()

    cen_poly = [
        (cen_start, lower_anchor),
        (cen_start, height),
        (cen_end, lower_anchor),
        (cen_end, height),
    ]
    chr_end_with_curve = chr_end + chr_len * curve
    chr_start_with_curve = chr_start - chr_len * curve
    # Define and draw the chromosome outline, taking into account the shape around the centromere
    outline = [
        (MplPath.MOVETO, (chr_start, height)),
        # Top left, bottom right: ‾\_
        (MplPath.LINETO, (cen_start, height)),
        (MplPath.LINETO, (cen_end, lower_anchor)),
        (MplPath.LINETO, (chr_end, lower_anchor)),
        # Right telomere: )
        (MplPath.LINETO, (chr_end, lower_anchor)),
        (MplPath.CURVE3, (chr_end_with_curve, ymid)),
        (MplPath.LINETO, (chr_end, height)),
        # Top right, bottom left: _/‾
        (MplPath.LINETO, (cen_end, height)),
        (MplPath.LINETO, (cen_start, lower_anchor)),
        (MplPath.LINETO, (chr_start, lower_anchor)),
        # Left telomere: (
        (MplPath.CURVE3, (chr_start_with_curve, ymid)),
        (MplPath.LINETO, (chr_start, height)),
        (MplPath.CLOSEPOLY, (chr_start, height)),
    ]
    if vertical == Orientation.VERTICAL:
        outline = [(command, coords[::-1]) for command, coords in outline]
        cen_poly = [coords[::-1] for coords in cen_poly]
    cen_patch = PathPatch(MplPath(cen_poly), facecolor=(0.8, 0.4, 0.4), lw=0, alpha=1, zorder=2)
    ax.add_patch(cen_patch)

    chr_move, chr_poly = zip(
        *outline,
        strict=True,
    )
    chr_patch = PathPatch(MplPath(chr_poly, chr_move), fill=None, joinstyle="round", alpha=1, zorder=2)
    ax.add_patch(chr_patch)

        
    # If start and stop positions are provided, draw a rectangle to highlight this region
    if start is not None and stop is not None:
        if zoom:
            # Zoom in on the specified region
            if vertical == Orientation.HORIZONTAL:
                ax.set_xlim(start, stop)
            else:
                ax.set_ylim(start, stop)
        else:
            if vertical == Orientation.HORIZONTAL:
                r = Rectangle(
                    (start, lower_anchor - target_region_extent),
                    width=stop - start,
                    height=height + 2 * target_region_extent,
                    fill=False,
                    edgecolor="r",
                    linewidth=1,
                    joinstyle="round",
                )
            else:
                r = Rectangle(
                    (lower_anchor - target_region_extent, start),
                    width=height + 2 * target_region_extent,
                    height=stop - start,
                    fill=False,
                    edgecolor="r",
                    linewidth=1,
                    joinstyle="round",
                )
            ax.add_patch(r)
    else:
        if not relative:
            if vertical == Orientation.HORIZONTAL:
                ax.set_xlim(chr_start_with_curve if lims_on_curve else chr_start, chr_end_with_curve if lims_on_curve else chr_end)
            else:
                ax.set_ylim(chr_start_with_curve if lims_on_curve else chr_start, chr_end_with_curve if lims_on_curve else chr_end)

    if regions:
        for r_start, r_stop, r_colour in regions:
            x0, width = r_start, r_stop - r_start
            y0 = lower_anchor + 0.02
            # print(f"x0 {x0}, width {width}, height: he")
            if vertical == Orientation.VERTICAL:
                x0 = lower_anchor + 0.03
                y0 = r_start
                height = width
                width = 0.94

            r = Rectangle(
                (x0, y0),  # +0.01 should shift us off outline of chromosome
                width=width,
                height=height,
                fill=kwargs.get("fill", True),
                color=r_colour,
                joinstyle="round",
                zorder=3,
                alpha=kwargs.get("alpha", 0.5),
                lw=kwargs.get("lw", 1),
            )
            ax.add_patch(r)
    if adjust_margins:
        # Adjust x-axis margins
        set_xmargin(ax, left=left_margin, right=right_margin)
        ax.set_ymargin(y_margin)

    # Remove axis spines and ticks for a cleaner look
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    # Add chromosome name to the plot
    if y_label is not None:
        if vertical == Orientation.VERTICAL:
            y0, _y1 = ax.get_ylim()
            ax.text(0.5, y0, y_label, fontsize=kwargs.get("fontsize", "x-large"), va="bottom", ha="center", rotation=90)
        else:
            x0, _x1 = ax.get_xlim()
            ax.text(x0, 1, y_label, fontsize=kwargs.get("fontsize", "x-large"), va="bottom")

    return ax

def make_ideogram_grid(
    target: str, 
    genome: GENOME = GENOME.HG38,
    start: int | None = None,
    stop: int | None = None,
    num_subplots=1, 
    subplot_width=3, 
    height_ratio = 0.5,
    ideogram_factor:float = 0.1,
    **ideogram_kwargs):
    """
    Create a grid of subplots, with an ideogram at the bottom. Meant to plot multiple features on the same chromosome.
    :param target: Target chromosome to plot.
    :param genome: Genome variant to use.
    :param start: Starting base pair position for the region of interest (optional). If start is None, stop must also be None.
    :param stop: Ending base pair position for the region of interest (optional). If stop is None, start must also be None.
    :param num_subplots: Number of subplots to create.
    :param subplot_width: Width of each subplot.
    :param height_ratio: Height ratio for the subplots.
    :param ideogram_factor: Height factor for the ideogram.
    :param ideogram_kwargs: Additional keyword arguments for the ideogram plotting function.
    :return: A tuple containing the figure and a list of axes for the subplots.
    """
    pfactor = int(1/ideogram_factor)
    gspec = gs(pfactor * num_subplots + 2 * num_subplots,1)
    fig = plt.figure(figsize=(subplot_width, subplot_width * height_ratio * num_subplots), facecolor="white")
    axes = []
    for i in range(num_subplots):
        ax = fig.add_subplot(gspec[pfactor * i + i: pfactor * (i + 1) + i, 0])
        axes.append(ax)
    for ax in axes[1:]:
        ax.sharex(axes[0])
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_xlabel("")
    ideogram_kwargs.update({
        "target": target,
        "genome": genome,
        "start": start,
        "stop": stop,
        "zoom": start is not None and stop is not None,
        "relative": ideogram_kwargs.get("relative", False),
        "adjust_margins": False,
        "lims_on_curve" : False,
    })
    ideogram_ax = plot_ideogram(fig.add_subplot(gspec[-num_subplots:, 0], sharex=axes[0]), **ideogram_kwargs)
    for obj in ideogram_ax.get_children():
        if hasattr(obj, "set_clip_on"):
            obj.set_clip_on(False)
    for ax in axes:
        ax.set_xlim(ideogram_ax.get_xlim())
    return fig, axes, ideogram_ax
    


if __name__ == "__main__":
    fig, axes = plt.subplots(
        ncols=1,
        nrows=22,
        figsize=(11, 11),
        facecolor="white",
    )
    genome = GENOME.CHM13
    for ax, contig_name in zip(axes, range(1, 23), strict=False):
        chromosome = f"chr{contig_name}"
        plot_ideogram(ax, target=chromosome, genome=genome)
    fig.savefig("ideogram.png", dpi=300)
