#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from java import jclass

import argparse
import  cv2, os
from removebg import RemoveBg


def doRemoveBG(img_folder, filename):
    # Removing backaground.

    rmbg = RemoveBg("hhPo6ZUVV2WfHM1TgBVbFGjo", "error.log")
    rmbg.remove_background_from_img_file(os.path.join(img_folder, filename))

    #cv2.imwrite(os.path.join(img_folder, filename)+'idf.jpg', image_original)
    return 0