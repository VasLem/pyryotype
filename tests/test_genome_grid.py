from pathlib import Path

from pyryotype import make_genome_grid

OUT_DIR = Path(__file__).parent.parent / "example_outputs"

def test_genome_grid():
    fig, axes, genome_ax = make_genome_grid(
        target_start="chr1",
        target_stop="chr5",
        num_subplots=2,
    )
    fig.savefig(OUT_DIR / "testing_genome_grid.png", bbox_inches="tight")