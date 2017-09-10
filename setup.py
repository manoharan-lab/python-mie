from setuptools import setup

setup(name='pymie',
      version='0.1',
      description='Pure Python package for Mie scattering calculations.',
      url='https://github.com/manoharan-lab/python-mie',
      author='Manoharan Lab, Harvard University',
      author_email='vnm@seas.harvard.edu',
      packages=['pymie'],
      install_requires=['pint', 'numpy', 'scipy'])
