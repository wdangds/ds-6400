# DS 6400

This is a Quarto book repository for weekly class writing.

## Environment

The Quarto project uses the `ds-6400` Jupyter kernel. That kernel should point to the conda environment named `ds-6400`, so notebooks and Quarto renders use the same Python and the same installed libraries.

Create the environment:

```bash
conda env create -f environment.yml
```

If the environment already exists and `environment.yml` changes, update it:

```bash
conda env update -f environment.yml --prune
```

Activate it for terminal work:

```bash
conda activate ds-6400
```

Register or refresh the matching Jupyter kernel:

```bash
python -m ipykernel install --user --name ds-6400 --display-name "Python (ds-6400)"
```

Optionally install the local `src/ds6400` package in editable mode:

```bash
python -m pip install -e .
```

The shared code is organized by purpose:

```text
src/ds6400/
  data_generation.py  # data-generating processes
  models.py           # model factory
  evaluation.py       # error estimators and CV helpers
  simulation.py       # Monte Carlo loops
  cache.py            # result cache paths
  progress.py         # logs and warning filters
```

Test the setup:

```bash
quarto render scratch/env-check.qmd
```

## Quarto

Preview the book:

```bash
quarto preview
```

Render the book:

```bash
quarto render
```

This renders HTML by default. To build the PDF version separately:

```bash
quarto render --profile pdf --to pdf
```

For the week 1 simulation, progress is written to `simulation-progress.log` instead of the rendered page. You can watch it in another terminal with:

```bash
tail -f simulation-progress.log
```

The extended simulation cache filename includes the sample-size grid. If you change the second `SAMPLE_SIZES` list in `weeks/01.qmd`, Quarto will request a new combined CSV instead of silently reusing the old one. Existing per-setting cache files may still be reused for sample sizes already computed.
