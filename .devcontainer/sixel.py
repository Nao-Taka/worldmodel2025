'''
このファイルをimportして各関数を使う
sprint:you can set any objects. this print PIL.Image, matplotlib.pyplot
sixel :you can convert Image to sixel(string)
'''
from io import BytesIO
import math
import os
from PIL import Image
import sys
import types

import numpy as np
import matplotlib.pyplot as plt

from libsixel import *

def sprint(*objs, scale=1.0, nCols=1):
    objs = list(objs)
    nImage = len(objs)

    #convert plot to Image
    for i in range(nImage):
        if isinstance(objs[i], Image.Image):
            pass
        elif isinstance(objs[i], types.ModuleType) and objs[i].__name__== 'matplotlib.pyplot':
            #convert plot to image
            objs[i] = __conv_plt2img(objs[i])
        else:
            print(type(objs[i]) , 'is not supported')
            objs[i] = None
    #remove none
    objs = [x for x in objs if x is not None]
    nImage = len(objs)

    #分割数
    img_rows = []
    for i in range(0, nImage, nCols):
        img_rows.append(objs[i:i+nCols])

    for img_row in img_rows:
        #それぞれの行ごとに幅の合計と最大高さを取得
        sum_width = 0
        max_height= 0
        for img in img_row:
            width, height = img.size
            #resize image
            if scale != 1.0:
                width, height = img.size
                width = int(width * scale)
                height = int(height * scale)
            sum_width += width
            max_height = max(max_height, height)

        #合成してsprintする
        #画像の結合
        new_image = Image.new('RGB', (sum_width, max_height))
        
        # 画像を横に並べて貼り付け
        x_offset = 0
        for img in img_row:
            #resize image
            if scale != 1.0:
                width, height = img.size
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = img.resize((new_width, new_height), resample=Image.LANCZOS)
            new_image.paste(img.convert('RGB'), (x_offset, 0))  # x_offsetの位置に画像を貼り付け
            x_offset += img.width  # 次の画像の位置を更新
        print(sixel(new_image))


def sixel(obj, scale=1.0):
    '''
    support:
    PIL.Image.Image
    matplotlib.pyplot(types.ModuleType)
    '''
    ret = 'no support'
    if isinstance(obj, Image.Image):
        ret = __img2sixel(obj, scale)
    elif isinstance(obj, types.ModuleType) and obj.__name__== 'matplotlib.pyplot':
        ret = __plt2sixel(obj, scale)
    return ret


def __img2sixel(image:Image.Image, scale=1.0):
    '''
    Input:PIL.Image.Image
    Output:Sixel text
    (you can view by using 'print')
    '''
    ret='0'
    
    #resize image
    if scale != 1.0:
        width, height = image.size
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), resample=Image.LANCZOS)

    s = BytesIO()
    width, height = image.size
    try:
        data = image.tobytes()
    except NotImplementedError:
        data = image.tostring()
    output = sixel_output_new(lambda data, s: s.write(data), s)
    try:
        if image.mode == 'RGBA':
            dither = sixel_dither_new(256)
            sixel_dither_initialize(dither, data, width, height, SIXEL_PIXELFORMAT_RGBA8888)
        elif image.mode == 'RGB':
            dither = sixel_dither_new(256)
            sixel_dither_initialize(dither, data, width, height, SIXEL_PIXELFORMAT_RGB888)
        elif image.mode == 'P':
            palette = image.getpalette()
            dither = sixel_dither_new(256)
            sixel_dither_set_palette(dither, palette)
            sixel_dither_set_pixelformat(dither, SIXEL_PIXELFORMAT_PAL8)
        elif image.mode == 'L':
            dither = sixel_dither_get(SIXEL_BUILTIN_G8)
            sixel_dither_set_pixelformat(dither, SIXEL_PIXELFORMAT_G8)
        elif image.mode == '1':
            dither = sixel_dither_get(SIXEL_BUILTIN_G1)
            sixel_dither_set_pixelformat(dither, SIXEL_PIXELFORMAT_G1)
        else:
            raise RuntimeError('unexpected image mode')
        try:
            sixel_encode(data, width, height, 1, dither, output)
            ret =  ((s.getvalue().decode('ascii')))
        finally:
            sixel_dither_unref(dither)
    finally:
        sixel_output_unref(output)
    return ret


def __plt2sixel(plot:plt, scale=1.0):
    '''
    Input:matplotlib.pyplot
    Output:Sixel text
    (you can view by using 'print')
    '''
    img = __conv_plt2img(plot)
    return __img2sixel(img, scale)

def __conv_plt2img(plot:plt):
    buf = BytesIO()
    plot.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    return img


if __name__=='__main__':
    print('this make use sixel easily')
    img = Image.open(os.path.dirname(os.path.abspath(__file__)) + '/apple.jpg')
    img_c = Image.open(os.path.dirname(os.path.abspath(__file__)) + '/coffee.jpg')
    sprint('apple.jpg')
    sprint(img,img,nCols=2, scale=0.5)
    # sprint(img, scale=3)
    plt.plot([1,2,3], [1,4,9])
    sprint(plt)
    # sprint(plt, 0.5)
    sprint(img, plt, img,scale= 0.3, nCols=4)
    sprint(img, img_c,'test',  plt, 'test', scale=0.6, nCols=2)
