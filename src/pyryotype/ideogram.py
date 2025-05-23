"""Ideogram plotting executables."""
from enum import Enum
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.axes._secondary_axes import SecondaryAxis
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.typing import ColorType
from matplotlib.text import Text
from matplotlib.gridspec import SubplotSpec
from matplotlib.gridspec import GridSpec as gs
from matplotlib.gridspec import GridSpecFromSubplotSpec as gsFromSubplotSpec

from pyryotype.plotting_utils import set_xmargin
import re
from typeguard import check_type
import numpy as np
from typing import Dict, Union, List, Literal

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
    lower_anchor: int = 0,
    height: int = 1,
    curve: float = 0.02,
    y_margin: float = 0.05,
    right_margin: float = 0.005,
    left_margin: float = 0.25,
    target_region_extent: float = 0.3,
    label: str | None = None,
    label_placement: Literal["height", "length"] = "height",
    label_kwargs: dict = None,
    vertical: Orientation = Orientation.HORIZONTAL,
    regions: list[tuple[int, int, ColorType]] | None = None,
    cytobands_df: pd.DataFrame = None,
    cytobands: Detail = Detail.CYTOBAND,
    relative: bool = True,
    adjust_margins: bool = True,
    _arrange_absolute_ax_lims: bool = True,
    **kwargs,    
):
    """
    Plot a chromosome ideogram with cytobands and optionally highlight a specific region.

    :param ax: Matplotlib axis object where the ideogram will be plotted.
    :param target: target chromosome to filter and plot.
    :param start: Starting base pair position for the region of interest (optional).
    :param stop: Ending base pair position for the region of interest (optional).
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
    :param cytobands_df: DataFrame containing cytoband data with columns "chrom", "chromStart",
      "chromEnd", "gieStain", and "colour".
    :param cytobands: Whether to render cytobands
    :param relative: Whether to plot the ideogram in relative coordinates (start of chromosome is 0) (default: True if single chromosome, False if target_stop!=target_start).
    :param adjust_margins: Whether to adjust the margins of the plot (default: True).

    :return: Updated axis object with the plotted ideogram.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax = plot_ideogram(ax, "chr1", start=50, stop=250, label="Chromosome 1")
    >>> ax.get_xlim()  # To test if the ideogram was plotted (not a direct measure but gives an idea)
    (-71574971.325, 256487353.7655)

    # Test behaviour with a non-existent chromosome
    >>> ax = plot_ideogram(ax, "chr_1", start=50, stop=250, label="Chromosome 1")# doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: Chromosome chr_1 not found in cytoband data. Should be one of ...

    """
    if label_kwargs is None:
        label_kwargs = dict()
    # some checks for input before we start
    if label is not None:
        assert label_placement in ["height", "length"], "label_placement must be either 'height' or 'length'"
    if start is not None and stop is not None:        
        assert start < stop, "Start must be less than stop"
    if regions is not None:
        assert isinstance(regions, list), "Regions must be a list of tuples"
        for region in regions:
            assert len(region) == 3, "Each region must be a tuple of (start, stop, colour)"
            assert isinstance(region[0], int), "Start must be an integer"
            assert isinstance(region[1], int), "Stop must be an integer"
            assert region[0] < region[1], "Start must be less than stop"
            assert check_type(region[2], ColorType), "Third element must be a colour"
    if cytobands_df is None:
        df = get_cytoband_df(genome, relative=relative)
    else:
        df = cytobands_df
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

    cen_outline = [
        (MplPath.MOVETO,(cen_start, lower_anchor)),
        (MplPath.LINETO, (cen_start, height)),
        (MplPath.LINETO, ((cen_start + cen_end)/2, ymid)),
        (MplPath.LINETO,(cen_end, height)),
        (MplPath.LINETO,(cen_end, lower_anchor)),
        (MplPath.LINETO, ((cen_start + cen_end)/2, ymid)),
        (MplPath.CLOSEPOLY,(cen_start, lower_anchor)),
    ]
    chr_end_without_curve = chr_end - chr_len * curve
    chr_start_without_curve = chr_start + chr_len * curve
    # Define and draw the chromosome outline, taking into account the shape around the centromere
    outline = [
        (MplPath.MOVETO, (chr_start_without_curve, height)),
        # Top part
        (MplPath.LINETO, (cen_start, height)),
        (MplPath.LINETO, ((cen_start + cen_end)/2, ymid)),
        (MplPath.LINETO, (cen_end, height)),
        (MplPath.LINETO, (chr_end_without_curve, height)),
        (MplPath.CURVE3, (chr_end, height)),
        (MplPath.CURVE3, (chr_end, ymid)),
        # Bottom part
        (MplPath.CURVE3, (chr_end, lower_anchor)),
        (MplPath.CURVE3, (chr_end_without_curve, lower_anchor)),
        (MplPath.LINETO, (chr_end_without_curve, lower_anchor)),
        (MplPath.LINETO, (cen_end, lower_anchor)),
        (MplPath.LINETO, ((cen_start + cen_end)/2, ymid)),
        (MplPath.LINETO, (cen_start, lower_anchor)),
        (MplPath.LINETO, (chr_start_without_curve, lower_anchor)),
        (MplPath.CURVE3, (chr_start, lower_anchor)),
        (MplPath.CURVE3, (chr_start, ymid)),
        (MplPath.CURVE3, (chr_start, height)),
        (MplPath.CURVE3, (chr_start_without_curve, height)),
        (MplPath.MOVETO, (chr_start_without_curve, height))
    ]
    def invert_with_curve(outline):
        outline = outline[::-1]
        new_outline = outline.copy()
        i = 0
        while i < len(outline):
            if outline[i][0] == MplPath.CURVE3:
                j = i + 1
                while j < len(outline) and outline[j][0] == MplPath.CURVE3:
                    j += 1
                new_outline[i:j] = outline[i+1:j] + [(MplPath.CURVE3, outline[j][1])] 
                i = j + 1
            else:
                i += 1
        return new_outline
    
    outside_outline = [(MplPath.MOVETO,(chr_start, height)),
                       (MplPath.LINETO,(chr_end, height)),
                       (MplPath.LINETO,(chr_end, lower_anchor)),
                       (MplPath.LINETO,(chr_start, lower_anchor)),
                       (MplPath.CLOSEPOLY,(chr_start, lower_anchor))] + invert_with_curve(outline)
    if vertical == Orientation.VERTICAL:
        outline = [(command, coords[::-1]) for command, coords in outline]
        cen_outline = [(command, coords[::-1]) for command, coords in cen_outline]
        outside_outline = [(command, coords[::-1]) for command, coords in outside_outline]
    cen_move, cen_poly = zip(
        *cen_outline,
        strict=True,
    )
    cen_patch = PathPatch(MplPath(cen_poly, cen_move), facecolor=(0.8, 0.4, 0.4), lw=0, alpha=1, zorder=2)
    
    ax.add_patch(cen_patch)

    chr_move, chr_poly = zip(
        *outline,
        strict=True,
    )
    mask_move, mask_poly = zip(
        *outside_outline,
        strict=True,
    )
    mask_patch = PathPatch(MplPath(mask_poly, mask_move), facecolor=(1.0, 1.0, 1.0), alpha=1, edgecolor=(1.0, 1.0, 1.0), zorder=2)
    ax.add_patch(mask_patch)
    chr_patch = PathPatch(MplPath(chr_poly, chr_move), fill = None, joinstyle="round", alpha=1, zorder=2)
    ax.add_patch(chr_patch)
        
    # If start and stop positions are provided, draw a rectangle to highlight this region
    if start is not None or stop is not None:
        if start is None:
            start = ax.get_xlim()[0]
        if stop is None:
            stop = ax.get_xlim()[1]
        # Zoom in on the specified region
        if vertical == Orientation.HORIZONTAL:
            ax.set_xlim(start, stop)
        else:
            ax.set_ylim(start, stop)
    else:
        if not relative and _arrange_absolute_ax_lims:
            if vertical == Orientation.HORIZONTAL:
                ax.set_xlim(chr_start, chr_end)
            else:
                ax.set_ylim(chr_start, chr_end)

    if regions:
        for r_start, r_stop, r_colour in regions:
            x0, rwidth = r_start, r_stop - r_start
            y0 = lower_anchor + 0.02
            # print(f"x0 {x0}, width {width}, height: he")
            rheight = height
            if vertical == Orientation.VERTICAL:
                x0 = lower_anchor + 0.03
                y0 = r_start
                rheight = rwidth
                rwidth = 0.94

            r = Rectangle(
                (x0, y0),  # +0.01 should shift us off outline of chromosome
                width=rwidth,
                height=rheight,
                fill=kwargs.get("fill", True),
                color=r_colour,
                joinstyle="round",
                zorder=3,
                alpha=kwargs.get("alpha", 0.5),
                lw=kwargs.get("lw", 1),
            )
            ax.add_patch(r)
    if vertical == Orientation.VERTICAL:
        ax.set_xlim(lower_anchor -0.05, height + 0.05)
    else:
        ax.set_ylim(lower_anchor -0.05, height + 0.05)
    if adjust_margins:
        # Adjust x-axis margins
        set_xmargin(ax, left=left_margin, right=right_margin)
        ax.set_ymargin(y_margin)

    # Remove axis spines and ticks for a cleaner look
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    def get_secondary_axis(ax, which: str):
        for x in ax.get_children():
            if isinstance(x, SecondaryAxis):
                if which == "x" and x._loc=="bottom":
                    return x
                if which == "y" and x._loc=="left":
                    return x
        if which == "x":
            return ax.secondary_xaxis("bottom")
        else:
            return ax.secondary_yaxis("left")
        
    # Add chromosome name to the plot
    if label is not None:
        if label_placement == "height":
            to_place = height/2
            if vertical == Orientation.VERTICAL:
                sec = get_secondary_axis(ax, "x")
                labs = sec.get_xticklabels()
                locs = sec.get_xticks()
            else:
                sec = get_secondary_axis(ax, "y")
                labs = sec.get_yticklabels()
                locs = sec.get_yticks()
        elif label_placement == "length":
            to_place = (chr_start + chr_end) / 2
            if vertical == Orientation.VERTICAL:
                sec = get_secondary_axis(ax, "y")
                labs = sec.get_yticklabels()
                locs = sec.get_yticks()
            else:
                sec = get_secondary_axis(ax, "x")
                labs = sec.get_xticklabels()
                locs = sec.get_xticks()
        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False
        tk = [i for i, (l,x) in enumerate(zip(labs,locs)) if not is_number(l.get_text()) or round(float(x), 2) != round(float(l.get_text()),2)]
        labs = [labs[i] for i in tk]
        locs = [locs[i] for i in tk]
            
        x = [i for i,(l,u) in enumerate(zip(locs[:-1],locs[1:])) if to_place  > l and to_place <= u]
        if x:
           pos = x[0]
        else:
            if locs and to_place > locs[-1]:
                pos = len(locs)
            else:
                pos = 0
        locs.insert(pos, to_place)
        labs.insert(pos, label)
    
        if label_placement == "height":
            if vertical == Orientation.VERTICAL:
                sec.set_xticks(locs, labs)
                if label_kwargs:
                    plt.setp(sec.get_xticklabels()[pos], **label_kwargs)
                sec.spines["bottom"].set_visible(False)
            else:
                sec.set_yticks(locs, labs)
                if label_kwargs:
                    plt.setp(sec.get_yticklabels()[pos], **label_kwargs)
                sec.spines["left"].set_visible(False)
        else:
            if vertical == Orientation.VERTICAL:
                sec.set_yticks(locs, labs)
                if label_kwargs:
                    plt.setp(sec.get_yticklabels()[pos], **label_kwargs)
                sec.spines["left"].set_visible(False)
            else:
                sec.set_xticks(locs, labs)
                if label_kwargs:
                    plt.setp(sec.get_xticklabels()[pos], **label_kwargs)
                sec.spines["bottom"].set_visible(False)
        sec.tick_params(axis=u'both', which=u'both',length=0)
        
    return ax

def _make_target_grid(
    
    target: Union[str, List[str]], 
    target_stop: str = None,
    genome: GENOME = GENOME.HG38,
    start: int | None = None,
    stop: int | None = None,
    num_subplots=1, 
    subplot_width=3, 
    height_ratio = 0.5,
    ideogram_factor:float = 0.1,
    fig: plt.Figure =None,
    subplot_spec: SubplotSpec=None,
    **ideogram_kwargs) -> tuple[plt.Figure, list[Axes], Axes]:
    """
    Create a grid of subplots, with an ideogram at the bottom. Meant to plot multiple features on the same chromosome.
    :param target: (Starting) target chromosome to filter and plot.
    :param target_stop: Ending target chromosome to filter and plot (optional).
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
    if target_stop is None:
        target_stop = target
    relative = ideogram_kwargs.get("relative", None)
    if relative is None:
        relative = target == target_stop
    cytobands_df = None
    chr_start = None
    chr_end = None
    if target != target_stop:
        cytobands_df = get_cytoband_df(genome, relative=False)
        chr_names = cytobands_df["chrom"].unique()
        chr_names = sorted(chr_names, key=chr_to_ord)
        targets = chr_names[chr_names.index(target): chr_names.index(target_stop) + 1]
        cytobands_df = cytobands_df[cytobands_df['chrom'].isin(targets)]
        chr_start = cytobands_df['chromStart'].min()
        chr_end = cytobands_df['chromEnd'].max()
        if relative:
            cytobands_df['chromEnd'] = cytobands_df['chromEnd'] - cytobands_df['chromStart'].min()
            cytobands_df['chromStart'] = cytobands_df['chromStart'] - cytobands_df['chromStart'].min()
    else:
        targets = [target]
    pfactor = int(1/ideogram_factor)
    
    if fig is None:
        fig = plt.figure(figsize=(subplot_width, subplot_width * height_ratio * num_subplots), facecolor="white")
    if subplot_spec is None:
        gspec = gs(pfactor * num_subplots + 2 * num_subplots -1, 1, hspace=0.05, wspace=0.05)
    else:
        gspec = gsFromSubplotSpec(pfactor * num_subplots + 2 * num_subplots - 1, 1, subplot_spec=subplot_spec, hspace=0.05, wspace=0.05)
    axes = []
    for i in range(num_subplots):
        ax = fig.add_subplot(gspec[pfactor * i + i: pfactor * (i + 1) + i, 0])
        axes.append(ax)
    for ax in axes[1:]:
        ax.sharex(axes[0])
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xlabel("")
    ideogram_ax = fig.add_subplot(gspec[-num_subplots:, 0], sharex=axes[0])
    ideogram_ax.set_xticks([])
    ideogram_ax.set_xticklabels([])
    ideogram_ax.set_xlabel("")
    for cnt, target in enumerate(targets):
        ideogram_kwargs.update({
            "target": target,
            "genome": genome,
            "label": target,
            "label_placement": ideogram_kwargs.get("label_placement", "height" if len(targets) == 1 else "length"),
            "start": None,
            "stop": None,
            "relative": ideogram_kwargs.get("relative", False),
            "adjust_margins": False,
        })
        ideogram_ax = plot_ideogram(ideogram_ax, cytobands_df=cytobands_df, _arrange_absolute_ax_lims=False, **ideogram_kwargs)
    if start is None:
        start = chr_start
    if stop is None:
        stop = chr_end
    ax.set_xlim(start, stop) 
    # for obj in ideogram_ax.get_children():
    #     if hasattr(obj, "set_clip_on"):
    #         obj.set_clip_on(False)
    for ax in axes:
        ax.set_xlim(ideogram_ax.get_xlim())
    axes[-1].spines["bottom"].set_visible(False)
    return fig, axes, ideogram_ax
    
def make_ideogram_grid(
    target: Union[str, List[str]], 
    genome: GENOME = GENOME.HG38,
    start: Union[str, Dict[str, int]] | None = None,
    stop: Union[str, Dict[str, int]] | None = None,
    num_subplots=1, 
    subplot_width=3, 
    height_ratio = 0.5,
    ideogram_factor:float = 0.1,
    grid_params: dict = None,
    **ideogram_kwargs) -> tuple[plt.Figure, Dict[str, list[Axes]], Dict[str, Axes]]:
    """
    Create a grid of subplots, with an ideogram at the bottom. Meant to plot multiple features on the same chromosome.
    :param target: Target chromosome(s) to plot.
    :param genome: Genome variant to use.
    :param start: Starting base pair position for the region of interest per target(optional). It must be a dictionary if multiple targets are provided. If start is not given for a target, stop must also not be given.
    :param stop: Ending base pair position for the region of interest per target(optional). It must be a dictionary if multiple targets are provided. If stop is not given for a target, start must also not be given.
    :param num_subplots: Number of subplots to create.
    :param subplot_width: Width of each subplot.
    :param height_ratio: Height ratio for the subplots.
    :param ideogram_factor: Height factor for the ideogram.
    :param ideogram_kwargs: Additional keyword arguments for the ideogram plotting function.
    :return: A tuple containing the figure and a list of axes for the subplots.
    """
    targets = target if isinstance(target, list) else [target]
    if grid_params is None:
        grid_params = dict()
    grid_params.update({"hspace": grid_params.get('hspace', 0.05),
                        "top": grid_params.get('top', 0.95),
                        "bottom": grid_params.get('bottom', 0.05),
                        "left": grid_params.get('left', 0.1),
                        "right": grid_params.get('right', 0.95),
                        })
    
    if len(targets) > 1:
        if isinstance(start, int):
            raise ValueError("If multiple targets are provided, start must be a dictionary")
        if isinstance(stop, int):
            raise ValueError("If multiple targets are provided, stop must be a dictionary")
    else:
        start = {targets[0]: start} if isinstance(start, int) else start
        stop = {targets[0]: stop} if isinstance(stop, int) else stop
    if start is None:
        start = dict()
    if stop is None:
        stop = dict()
    
    start = {t: start.get(t, None) for t in targets}
    stop = {t: stop.get(t, None) for t in targets}
    fig = plt.figure(figsize=(subplot_width, subplot_width * height_ratio * num_subplots * len(targets)), facecolor="white", )
    gs0 = gs(len(targets), 1, figure=fig, **grid_params)
    value_axes = {}
    ideogram_axes = {}
    for i, target in enumerate(targets):
        
        _, a0, a1 = _make_target_grid(
            target, 
            genome=genome,
            start=start[target], 
            stop=stop[target], 
            num_subplots=num_subplots, 
            subplot_width=subplot_width, 
            height_ratio=height_ratio, 
            ideogram_factor=ideogram_factor,
            subplot_spec=gs0[i, 0],
            relative=True,
            fig=fig, 
            **ideogram_kwargs)
        value_axes[target] = a0
        ideogram_axes[target] = a1
    fig.tight_layout()
    return fig, value_axes, ideogram_axes

def make_genome_grid(target_start: str, target_stop: str, genome: GENOME = GENOME.HG38, num_subplots=1, subplot_width=10, height_ratio = 0.5, 
                     ideogram_factor:float = 0.1, **ideogram_kwargs) -> tuple[plt.Figure, list[Axes], Axes]:
    """
    Create a grid of subplots for a specific genome with a specified start and stop target.
    :param target_start: Starting target, e.g., chr1.
    :param target_stop: Ending target, e.g., chr5.
    :param genome: Genome variant to use.
    :param num_subplots: Number of subplots to create.
    :param subplot_width: Width of each subplot.
    :param height_ratio: Height ratio for the subplots.
    :param ideogram_factor: Height factor for the ideogram.
    :param ideogram_kwargs: Additional keyword arguments for the ideogram plotting function.
    :return: A tuple containing the figure and a list of axes for the subplots.
    """
    fig, axes, genome_ax = _make_target_grid(
        target=target_start,
        target_stop=target_stop,
        genome=genome,
        num_subplots=num_subplots,
        subplot_width=subplot_width,
        height_ratio=height_ratio,
        ideogram_factor=ideogram_factor,
        **ideogram_kwargs
    )
    return fig, axes, genome_ax

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
