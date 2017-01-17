#!/usr/bin/env python

import sys
import os
import cgi
import json
import cgitb
cgitb.enable()

from PIL import Image
import numpy as np
from io import BytesIO
from binascii import a2b_base64

from mytensor import MyTensor

import logging

print('Content-Type: text/json; charset=utf-8')
print()

if os.environ['REQUEST_METHOD'] == 'POST':
    data = cgi.FieldStorage()
    img_str = data.getvalue('img', None)
    if img_str:
        b64_str = img_str.split(',')[1]
        img = Image.open(BytesIO(a2b_base64(b64_str))).convert('L')
        img_arr = np.array(img).reshape(1, -1)
        tf = MyTensor()
        result = tf.predict(img_arr)
        print(json.dumps({'status': True, 'num': result}))
        sys.exit()
print(json.dumps({'status': False, 'num': False}))