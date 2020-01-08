from setuptools import setup, find_packages

version = '0.4'

setup(name='pymetrikz',
      version=version,
      description="Visual Quality Assessment Package and Tools",
      classifiers=[],
      keywords='pymetrikz',
      author='Pedro Garcia Freitas',
      author_email='sawp@sawp.com.br',
      url='http://www.sawp.com.br/projects/pymetrikz',
      license='GNU GPL version 2',
      py_modules=['metrikz', 'pymetrikz'],
      namespace_packages=[],
      include_package_data=True,
      zip_safe=False,
      install_requires=[
          'scipy', 'numpy'
      ],
      entry_points={
          'console_scripts': [
              'pymetrikz = pymetrikz:__main',
              ],
          },
      )
