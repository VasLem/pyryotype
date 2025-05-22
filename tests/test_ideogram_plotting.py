from itertools import chain
from pathlib import Path

from matplotlib import pyplot as plt
from pyryotype import plot_ideogram, make_ideogram_grid, make_genome_grid
from pyryotype.ideogram import GENOME, Detail, Orientation

OUT_DIR = Path(__file__).parent.parent / "example_outputs"


def test_simple_vertical_chr1():
    fig, ax = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(3, 25),
        facecolor="white",
    )

    plot_ideogram(ax, target="chr1", left_margin=0, y_label="", vertical=Orientation.VERTICAL)

    fig.savefig(OUT_DIR / "testing_vert.png", bbox_inches="tight")
    
def test_simple_horizontal_chr1():
    fig, ax = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(25, 3),
        facecolor="white",
    )

    plot_ideogram(ax, target="chr1", left_margin=0, y_label="", vertical=Orientation.HORIZONTAL)

    fig.savefig(OUT_DIR / "testing_horz.png", bbox_inches="tight")
    

def test_simple_vertical_chr1_start_stop():
    fig, ax = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(3, 25),
        facecolor="white",
    )

    plot_ideogram(ax, target="chr1", left_margin=0, y_label="", vertical=Orientation.VERTICAL, start=150000, stop=50000000)

    fig.savefig(OUT_DIR / "testing_vert_start_stop.png", bbox_inches="tight")
    
    
def test_simple_horizontal_chr1_start_stop_zoom():
    fig, ax = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(25, 3),
        facecolor="white",
    )

    plot_ideogram(ax, target="chr1", left_margin=0, y_label="", vertical=Orientation.HORIZONTAL, start=150000, stop=50000000, zoom=True)

    fig.savefig(OUT_DIR / "testing_horz_start_stop_zoom.png", bbox_inches="tight")
    

def test_23_vertical_chm13():
    genome = GENOME.CHM13
    fig, axes = plt.subplots(ncols=24, nrows=1, figsize=(15, 25), facecolor="white", sharey=True)

    for ax, i in zip(axes, chain(range(1, 23), iter("XY")), strict=True):
        _ax = plot_ideogram(
            ax, target=f"chr{i}", y_label=f"Chr. {i}", left_margin=0, vertical=Orientation.VERTICAL, genome=genome
        )

    fig.savefig(OUT_DIR / "testing_vert_23.png", bbox_inches="tight")


def test_23_vertical_hg38():
    genome = GENOME.HG38
    fig, axes = plt.subplots(ncols=22, nrows=1, figsize=(15, 25), facecolor="white", sharey=True)

    for ax, i in zip(axes, chain(range(1, 23)), strict=True):
        _ax = plot_ideogram(
            ax, target=f"chr{i}", y_label=f"Chr. {i}", left_margin=0, 
            vertical=Orientation.VERTICAL, genome=genome, relative=True
        )

    fig.savefig(OUT_DIR / "testing_vert_23_hg38.png", bbox_inches="tight")


def test_23_vertical_hg19():
    genome = GENOME.HG19
    fig, axes = plt.subplots(ncols=22, nrows=1, figsize=(15, 25), facecolor="white", sharey=True)

    for ax, i in zip(axes, chain(range(1, 23)), strict=True):
        _ax = plot_ideogram(
            ax, target=f"chr{i}", y_label=f"Chr. {i}", left_margin=0, vertical=Orientation.VERTICAL, genome=genome
        )

    fig.savefig(OUT_DIR / "testing_vert_23_hg19.png", bbox_inches="tight")


def test_23_vertical_chm13_bare():
    genome = GENOME.CHM13
    fig, axes = plt.subplots(ncols=24, nrows=1, figsize=(15, 25), facecolor="white", sharey=True)

    for ax, i in zip(axes, chain(range(1, 23), iter("XY")), strict=True):
        _ax = plot_ideogram(
            ax,
            target=f"chr{i}",
            y_label=f"Chr. {i}",
            left_margin=0,
            vertical=Orientation.VERTICAL,
            cytobands=Detail.BARE,
            genome=genome,
        )

    fig.savefig(OUT_DIR / "testing_vert_bare_23.png", bbox_inches="tight")


def test_23_horizontal_chm13_bare():
    genome = GENOME.CHM13
    fig, axes = plt.subplots(ncols=1, nrows=24, figsize=(25, 15), facecolor="white", sharey=True)

    for ax, i in zip(axes, chain(range(1, 23), iter("XY")), strict=True):
        _ax = plot_ideogram(
            ax,
            target=f"chr{i}",
            y_label=f"Chr. {i}",
            left_margin=0,
            vertical=Orientation.HORIZONTAL,
            cytobands=Detail.BARE,
            genome=genome,
        )

    fig.savefig(OUT_DIR / "testing_horz_bare_23.png", bbox_inches="tight")


def test_23_vertical_chm13_regions():
    genome = GENOME.CHM13
    fig, axes = plt.subplots(ncols=24, nrows=1, figsize=(15, 25), facecolor="white", sharey=True)

    per_chr_regions = {"chr1": [(0, 1000000, "black"), (20_000_000, 35_000_000, "red")]}
    for ax, i in zip(axes, chain(range(1, 23), iter("XY")), strict=True):
        regions = per_chr_regions.get(f"chr{i}")
        _ax = plot_ideogram(
            ax,
            target=f"chr{i}",
            y_label=f"Chr. {i}",
            left_margin=0,
            vertical=Orientation.VERTICAL,
            genome=genome,
            regions=regions,
        )

    fig.savefig(OUT_DIR / "testing_vert_23_regions.png", bbox_inches="tight")


def test_23_horz_chm13_regions():
    genome = GENOME.CHM13
    fig, axes = plt.subplots(ncols=1, nrows=24, figsize=(15, 25), facecolor="white", sharey=True)

    per_chr_regions = {"chr1": [(0, 1000000, "black"), (20_000_000, 25_000_000, "red")]}
    for ax, i in zip(axes, chain(range(1, 23), iter("XY")), strict=True):
        regions = per_chr_regions.get(f"chr{i}", None)
        _ax = plot_ideogram(
            ax,
            target=f"chr{i}",
            y_label=f"Chr. {i}",
            left_margin=0,
            height=0.99,
            vertical=Orientation.HORIZONTAL,
            genome=genome,
            regions=regions,
        )

    fig.savefig(OUT_DIR / "testing_horz_23_regions.png", bbox_inches="tight")



def test_ideogram_grid_generation():
    fig, axes, ideogram_ax = make_ideogram_grid(
        target="chr1",
        num_subplots=2,
    )
    fig.savefig(OUT_DIR / "testing_ideogram_grid.png", bbox_inches="tight")

def test_ideogram_grid_generation_three_targets():
    fig, axes, ideogram_ax = make_ideogram_grid(
        target=["chr1", "chr2", "chr22"],
        num_subplots=2,
    )
    fig.savefig(OUT_DIR / "testing_ideogram_grid_three_targets.png", bbox_inches="tight")

def test_ideogram_grid_generation_three_targets_with_start_stop():
    # should raise
    try:
        fig, axes, ideogram_ax = make_ideogram_grid(
            target=["chr1", "chr2", "chr22"],
            start=0,
            stop=1,
            num_subplots=2,
        )
        raise ValueError("Should have raised an error")
    except ValueError as e:
        pass
    fig, axes, ideogram_ax = make_ideogram_grid(
        target=["chr1", "chr2", "chr22"],
        start={"chr1": 0, "chr2": 0, "chr22": 0},
        stop={"chr1": 50000000, "chr2": 25000000, "chr22": 1500000},
        zoom=True,
        num_subplots=2,
    )
    fig.savefig(OUT_DIR / "testing_ideogram_grid_three_targets_with_start_stop.png")

def test_genome_grid():
    fig, axes, genome_ax = make_genome_grid(
        target_start="chr1",
        target_stop="chr5",
        num_subplots=2,
    )
    fig.savefig(OUT_DIR / "testing_genome_grid.png", bbox_inches="tight")