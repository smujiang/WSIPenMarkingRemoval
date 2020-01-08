#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# pymetrikz - Tool for image metrics comparisons.
#
# Copyright (C) 2011 Pedro Garcia Freitas <sawp@sawp.com.br>
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
Compare two or more images using MSE, PSNR, SNR, SSIM, UQI, PBVIF, MSSIM,
NQM and WSNR metrics.

For usage and a list of options, try this:
$ ./pymetrikz -h

This program and its regression test suite live here:
http://www.sawp.com.br/projects/pymetrikz"""

__author__ = "Pedro Garcia Freitas <sawp@sawp.com.br>"
__version__ = "$Revision: 0.1 $"
__date__ = "$Date: 2011/12/04 $"
__copyright__ = "Copyright (c) 2011 Pedro Garcia"
__license__ = "GPLv2"

import metrikz
import re
from sys import argv, exit
from scipy.misc import imread, imsave


class CommandLineOpt:
    def __init__(self):
        self.isSelectedHelp = ('-h' in argv) or ('-help' in argv)
        self.isSelectedMSE = '-mse' in argv
        self.isSelectedPSNR = '-psnr' in argv
        self.isSelectedSNR = '-snr' in argv
        self.isSelectedSSIM = '-ssim' in argv
        self.isSelectedUQI = '-uqi' in argv
        self.isSelectedPBVIF = '-pbvif' in argv
        self.isSelectedMSSIM = '-mssim' in argv
        self.isSelectedNQM = '-nqm' in argv
        self.isSelectedWSNR = '-wsnr' in argv
        self.isSelectedRMSE = '-rmse' in argv
        self.__detect_if_all_false()
        self.isSelectedLatex = ('-latex' in argv) or ('-l' in argv)
        self.isSelectedHTML = ('-html' in argv) or ('-htm' in argv)

    def __detect_if_all_false(self):
        a = not self.isSelectedMSE and not self.isSelectedPSNR
        b = not self.isSelectedSSIM and not self.isSelectedUQI
        c = not self.isSelectedSNR and not self.isSelectedPBVIF
        d = not self.isSelectedMSSIM and not self.isSelectedNQM
        e = not self.isSelectedWSNR and not self.isSelectedRMSE
        test = a and b and c and d and e
        if test:
            self.isSelectedMSE = True
            self.isSelectedPSNR = True
            self.isSelectedSNR = True
            self.isSelectedSSIM = True
            self.isSelectedUQI = True
            self.isSelectedPBVIF = True
            self.isSelectedMSSIM = True
            self.isSelectedNQM = True
            self.isSelectedWSNR = True
            self.isSelectedRMSE = True


def get_usage():
    return "Usage: pymetrikz [options] reference.bmp query1.bmp query2.bmp ..."


def get_help():
    def __get_help_line():
        helpline = "  %-20s %s\n"
        return helpline

    def __get_help_comments():
        c = []
        c.append(('-h, -help', 'show this help message and exit'))
        c.append(('-l, -latex', 'print the metrics as LaTeX table.'))
        c.append(('-html, -html', 'print the metrics as HTML table.'))
        c.append(('-default', 'print the metrics as ASCII text table.'))
        c.append(('-mse', 'compute the mean square error metric.'))
        c.append(('-rmse', 'compute the root-mean-square error metric.'))
        c.append(('-psnr', 'compute the peak signal-to-noise ratio metric.'))
        c.append(('-snr', 'compute the signal-to-noise ratio metric.'))
        c.append(('-ssim', 'compute the structural similarity metric.'))
        c.append(('-uqi', 'compute the universal image quality index metric.'))
        c.append(('-pbvif', 'compute the visual information fidelity metric.'))
        c.append(('-mssim', 'compute the multi-scale SSIM index metric.'))
        c.append(('-nqm', 'compute the noise quality measure.'))
        c.append(('-wsnr', 'compute the weighted signal-to-noise ratio.'))
        return c

    def __get_options():
        c = __get_help_comments()
        options = ""
        helpline = __get_help_line()
        for cmm in c:
            options += helpline % cmm
        return options

    help = get_usage()
    help += "\n\n"
    help += "Options:\n"
    help += __get_options()
    help += "\n\n"
    help += "Documentation, examples and more information:\n"
    help += "        http://www.sawp.com.br/projects/pymetrikz"
    help += "\n\n"
    help += "Contacts to send bug reports, comments or feedback:\n"
    help += "        Pedro Garcia Freitas <sawp@sawp.com.br>"
    return help


def calculate_metrics(ref, query):
    metrics = {}
    if __cl.isSelectedMSE:
        metrics['mse'] = metrikz.mse(ref, query)
    if __cl.isSelectedRMSE:
        metrics['rmse'] = metrikz.rmse(ref, query)
    if __cl.isSelectedPSNR:
        metrics['psnr'] = metrikz.psnr(ref, query)
    if __cl.isSelectedSNR:
        metrics['snr'] = metrikz.snr(ref, query)
    if __cl.isSelectedSSIM:
        metrics['ssim'] = metrikz.ssim(ref, query)
    if __cl.isSelectedUQI:
        metrics['uqi'] = metrikz.uqi(ref, query)
    if __cl.isSelectedPBVIF:
        metrics['pbvif'] = metrikz.pbvif(ref, query)
    if __cl.isSelectedMSSIM:
        metrics['mssim'] = metrikz.mssim(ref, query)
    if __cl.isSelectedNQM:
        metrics['nqm'] = metrikz.nqm(ref, query)
    if __cl.isSelectedWSNR:
        metrics['wsnr'] = metrikz.wsnr(ref, query)
    return metrics


def grep_regex(string, regex):
    s = re.compile(regex)
    result = s.match(string)
    if result is not None:
        return result.group(0)
    return None


def get_image_types_regexes():
    types = ["\\w+.bmp"]
    types += ["\\w+.png"]
    types += ["\\w+.jpg"]
    types += ["\\w+.jpeg"]
    types += ["\\w+.pgm"]
    types += ["\\w+.gif"]
    return types


def get_images():
    images = []
    for regex in get_image_types_regexes():
        for im in argv:
            i = grep_regex(im, regex)
            if i is not None:
                images.append(i)
    return images


def check_image_availability(images):
    if len(images) <= 1:
        print "Error: Use 1 reference image and at least 1 query image."
        get_usage()
        exit()


def check_same_size(ref, img):
    if not (ref.shape == img.shape):
        print("Error: The images must have the same size.")
        exit()


def associate_metrics_with_images():
    images = get_images()
    check_image_availability(images)
    (ref, queries) = (images[0], images[1:])
    if ref in queries:
        queries.remove(ref)
    ref = imread(ref, flatten=True)
    output = {}
    for query in queries:
        q = imread(query, flatten=True)
        check_same_size(ref, q)
        output[query] = calculate_metrics(ref, q)
    return output


def create_ASCII_table():
    def __create_line_format():
        cols = count_cols()
        maxlen = __get_max_length()
        line = "|%-" + str(maxlen) + "s"
        for c in range(cols):
            line += "|%11s".format('centered')
        line += "|\n"
        return line

    def __get_max_length():
        image_names = image_metrics.keys()
        f = lambda x: len(x)
        image_sizes = map(f, image_names)
        return max(image_sizes)

    def __create_header():
        selected_metrics = image_metrics.values()[0]
        line = __create_line_format()
        l = ["image"] + selected_metrics.keys()
        header = line % tuple(l)
        return header

    def __create_lines():
        line = __create_line_format()
        lines = ""
        for img in image_metrics.iterkeys():
            metrics = image_metrics[img]
            f = lambda x: "%.5f" % x
            l = [img] + map(f, metrics.values())
            lines += line % tuple(l)
        return lines
    image_metrics = associate_metrics_with_images()
    header = __create_header()
    lines = __create_lines()
    table = header + lines
    return table


def create_LATEX_table():
    def __create_line_format():
        cols = count_cols()
        line = "%-13s &"
        for c in range(cols - 1):
            line += "%11s &"
        line += "%11s "
        line += "\\\ \n"
        return line

    def __create_first_line():
        image_metrics = associate_metrics_with_images()
        selected_metrics = image_metrics.values()[0]
        l = ["image"] + selected_metrics.keys()
        line_format = __create_line_format()
        first_line = line_format % tuple(l)
        return first_line

    def __create_header():
        cols = count_cols()
        fmt_cols = "".join(['|l' for i in range(cols)]) + "|"
        header = "\\begin{table}\n"
        header += "  \\begin{tabular}{" + fmt_cols + "}\n"
        header += "    \\hline\n"
        header += __create_first_line() + "\\hline\n"
        return header

    def __create_footer():
        footer = "  \\end{tabular}\n"
        footer += "\\end{table}\n"
        return footer

    def __create_lines():
        image_metrics = associate_metrics_with_images()
        line = __create_line_format()
        lines = ""
        for img in image_metrics.iterkeys():
            metrics = image_metrics[img]
            f = lambda x: "%.5f" % x
            l = [img] + map(f, metrics.values())
            lines += line % tuple(l)
        return lines

    header = __create_header()
    lines = __create_lines()
    footer = __create_footer()
    table = header + lines + footer
    return table


def create_HTML_table():
    def __create_line_format():
        cols = count_cols()
        line = "  <tr>\n    <td>%s</td>"
        for c in range(cols):
            line += "\n    <td>%s</td>"
        line += "\n  </tr>\n"
        return line

    def __create_first_line():
        image_metrics = associate_metrics_with_images()
        selected_metrics = image_metrics.values()[0]
        l = ["image"] + selected_metrics.keys()
        line_format = __create_line_format()
        first_line = line_format % tuple(l)
        return first_line

    def __create_header():
        cols = count_cols()
        header = "<table>\n"
        header += __create_first_line()
        return header

    def __create_footer():
        footer = "</table>\n"
        return footer

    def __create_lines():
        image_metrics = associate_metrics_with_images()
        line = __create_line_format()
        lines = ""
        for img in image_metrics.iterkeys():
            metrics = image_metrics[img]
            f = lambda x: "%.5f" % x
            l = [img] + map(f, metrics.values())
            lines += line % tuple(l)
        return lines

    header = __create_header()
    lines = __create_lines()
    footer = __create_footer()
    table = header + lines + footer
    return table


def select_output_type():
    if __cl.isSelectedHTML:
        return create_HTML_table()
    elif __cl.isSelectedLatex:
        return create_LATEX_table()
    else:
        return create_ASCII_table()


def count_cols():
    count = 0
    if __cl.isSelectedMSE:
        count += 1
    if __cl.isSelectedRMSE:
        count += 1
    if __cl.isSelectedPSNR:
        count += 1
    if __cl.isSelectedSNR:
        count += 1
    if __cl.isSelectedSSIM:
        count += 1
    if __cl.isSelectedUQI:
        count += 1
    if __cl.isSelectedPBVIF:
        count += 1
    if __cl.isSelectedMSSIM:
        count += 1
    if __cl.isSelectedNQM:
        count += 1
    if __cl.isSelectedWSNR:
        count += 1
    return count


def check_help():
    if __cl.isSelectedHelp:
        print get_help()
        exit(0)


def check_usage():
    if len(argv) < 2:
        print get_usage()
        exit(0)


def __main():
    check_help()
    check_usage()
    print select_output_type()


__cl = CommandLineOpt()


if __name__ == '__main__':
    __main()
