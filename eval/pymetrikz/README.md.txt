![image](https://bytebucket.org/kuraiev/pymetrikz/raw/48437fa969f9862c8951d9426bf292abc14206cd/logo.png)

Contents
========

1.  [Description](#description)
2.  [License](#license)
3.  [Requirements](#requeriments)
4.  [Installation](#installation)
5.  [Instructions](#instructions)
6.  [Examples](#examples)
7.  [Contact](#contact)
8.  [Contributing](#contributing)
9.  [References](#references)
10. [Thanks](#thanks)

Description
===========

PyMetrikz is a Python tool and module package that implements several
visual quality metrics algorithms. These algorithms are comparison
functions that differentiate the original image (reference) with a
modified version of the image (query). All algorithms implemented in
this package requires operate on two images of the same size. For
reference image and query image with distinct sizes, the metrics are out
of scope.

The purpose of this package is make a small library using the most used
metrics to compare images for Python language. Usually, these
implemented metrics are available in Matlab. Sometimes, It can be
awkward, because Matlab is a non-free, hard-integration and restrictive
software. Using PyMetrikz packages, the users can be avoid the
dependency of Matlab in open-sources projects.

All algorithms implemented **were not** developed or proposed by the
author of this package. Instead, the author of PyMetrikz just have
implemented these non-patented algorithms to allow the use in a easy
way. All authors of algorithms and them articles are properly cited in
"References" section of this document.

License
=======

PyMetrikz is released under [GNU GPL version
2.](http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt)

Requeriments
============

To run the PyMetrikz, Python 2.6 or later must be installed.

In addition, it requires the additional installation of Setuptools,
Python Image Library (PIL), NumPy and SciPy libraries.

Instalation
===========

1.  Clone the last version from the BitBucket repository.
> `# hg clone https://bitbucket.org/kuraiev/pymetrikz`
2.  Go to the directory where 'setup.py' is located.
> `# cd pymetrikz/pymetrikz/`
3.  Install using the command as root
> `# python setup.py install`

Instructions
============

You can use PyMetrikz as command-line application or call the functions
in your applications.

Using as application
--------------------

If PyMetrikz is installed in your system, the 'pymetrikz' command should
be available from shell. The options of this program is listed using the
`'-h'` or `'-help'` options. The usage example printed is like this:

If PyMetrikz is installed in your system, the `pymetrikz` command should be
available from shell. The options of this program is listed using
`'-h'` or `'-help'` options. The usage example printed is like this:

    [pedrogarcia@freebsd ~]$ pymetrikz -h
    
    Usage: pymetrikz [options] reference.bmp query1.bmp query2.bmp ...
    
    Options:

    -h, -help
    show this help message and exit.
    
    -l, -latex
    print the metrics as LaTeX table.
    
    -html, -htm
    print the metrics as HTML table.
    
    -default
    print the metrics as ASCII text table.
    
    -mse
    compute the mean square error metric.
    
    -psnr
    compute the peak signal-to-noise ratio metric.
    
    -snr
    compute the signal-to-noise ratio metric.
    
    -ssim
    compute the structural similarity metric.
    
    -uqi
    compute the universal image quality index metric.
    
    -pbvif
    compute the visual information fidelity metric.
    
    -mssim
    compute the multi-scale SSIM index metric.
    
    -nqm
    compute the noise quality measure.
    
    -wsnr
    compute the weighted signal-to-noise ratio.


The flags `-l`, `-latex`, `-html`, `-htm` and `-default` are formatting
options, exporting the results in a latex table, html table or raw ascii text,
respectively.

The `-mse`, `-psnr`, `-snr`, `-ssim`, `-uqi`, `-pbvif`, `-mssim`, `-nqm` and 
`-wsnr` flags insert a column with the correspondent metric in the output table.

Using as library
----------------

PyMetrikz can be used as third-part library for others projects. Python
programs that require the implemented metrics can import the `metrikz`
module and call its functions.

Examples
========

Using as application
--------------------

### Comparing two images (lena2.bmp and lena3.bmp as query, lena.bmp as reference) using MSE, PSNR and SSIM and exporting the result in LaTeX format

> ![image1](https://bytebucket.org/kuraiev/pymetrikz/raw/9f93f15e8e29bb7549b7f18079470c1cad05f711/i1.png)

### Comparing three images using UQI, PBVIF, NQM and WSNR and exporting to a HTML table

> ![image2](https://bytebucket.org/kuraiev/pymetrikz/raw/9f93f15e8e29bb7549b7f18079470c1cad05f711/i2.png)

### Comparing the reference image (lena.bmp) with all others images in the directory, using all metrics, and printing the result as ASCII text

> ![image3](https://bytebucket.org/kuraiev/pymetrikz/raw/9f93f15e8e29bb7549b7f18079470c1cad05f711/i3.png)

Using as library
----------------

### Code showing how to import the required libraries, read the reference image (a), read the query image (b) and use the metrics functions.

> ![image4](https://bytebucket.org/kuraiev/pymetrikz/raw/9f93f15e8e29bb7549b7f18079470c1cad05f711/i4.png)

Contact
=======

Please send all comments, questions, reports and suggestions (especially if 
you would like to contribute) to **sawp@sawp.com.br**

Contributing
============

If you would like to contribute with new algorithms, increment of code
performance, documentation or another kind of modifications, please
contact us. The only requirements are: keep the code compatible with
PEP8 standardization and licensed by GPLv2.

References
==========

**Structural Similarity Index (SSIM)**  
Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality 
assessment: From error visibility to structural similarity" IEEE Transactions 
on Image Processing, vol. 13, no. 4, pp.600-612, Apr. 2004

**Multi-scale SSIM Index (MSSIM)**  
Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality
assessment: From error visibility to structural similarity" IEEE Transactions
on Image Processing, vol. 13, no. 4, pp.600-612, Apr. 2004

**Noise Quality Measure (NQM)**  
N. Damera-Venkata, T. Kite, W. Geisler, B. Evans and A. Bovik, "Image Quality
Assessment Based on a Degradation Model", IEEE Trans. on Image Processing,
Vol. 9, No. 4, Apr. 2000

**Universal Image Quality Index (UQI)**  
Zhou Wang and Alan C. Bovik, "A Universal Image Quality Index", IEEE Signal
Processing Letters, 2001

**Visual Information Fidelity (VIF)**  
H. R. Sheikh and A. C. Bovik, "Image Information and Visual Quality"., IEEE
Transactions on Image Processing, (to appear).

**Weighted Signal-to-Noise Ratio (WSNR)**  
T. Mitsa and K. Varkur, "Evaluation of contrast sensitivity functions for the
formulation of quality measures incorporated in halftoning algorithms",
ICASSP '93-V, pp. 301-304.

**Signal-to-Noise Ratio (SNR, PSNR)**  
J. Mannos and D. Sakrison, "The effects of a visual fidelity criterion on the
encoding of images", IEEE Trans. Inf. Theory, IT-20(4), pp. 525-535, July 1974

Thanks
======

Special thanks to [Mylene C. Q. Farias][12].

Thanks to Philip Lacombe for suggestions about division by zero at PSNR
function.

Thanks to Peter Pakulski from [CiSRA][13] for reporting errors when using colored images.

[10]: http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt
[11]: http://www.sawp.com.br/projects/pymetrikz/files/pymetrikz-0.3.tar.gz
[12]: http://www.gpds.ene.unb.br/mylene/
[13]: http://www.cisra.com.au/about.html
