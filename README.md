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
