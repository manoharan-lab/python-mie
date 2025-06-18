# python-mie
Pure Python package for Mie scattering calculations. This package is used by,
for example, the [structural-color python
package](https://github.com/manoharan-lab/structural-color).

To install:

```shell
git clone https://github.com/manoharan-lab/python-mie.git
python -m pip install ./python-mie
```

To remove:

```shell
python -m pip uninstall pymie
```

You can then run the tests to make sure the package works properly:
```shell
cd python-mie
pytest -v
```

The original code was developed by Jerome Fung in the [Manoharan Lab at Harvard
University](http://manoharan.seas.harvard.edu). The code has since been updated
by Victoria Hwang, Anna B. Stephenson, and Vinothan N. Manoharan. It works
in Python 3, and it can handle quantities with dimensions (using
[pint](https://github.com/hgrecco/pint)).

The development of this code was made possible with support from the National
Science Foundation under award numbers CBET-0747625, DMR-0820484, DMR-1306410,
DMR-1420570, and DMR-2011754.
