# -*- coding: utf-8 -*-

from ai_makeup_predict import AIMakeupPredict
import os
import numpy as np
import setting
from dlib_makeup import Makeup

if __name__ =='__main__':
    image_url = ''
    mu = Makeup() # 实例化化妆器
    mouth_location = {} # 用于保存嘴巴位置

    chanel  = 3
    mouth_w = 90  # 嘴巴宽
    mouth_h = 30  # 嘴巴高

    ai_makeup = AIMakeupPredict()

    # 载入图片的处理
    while True:
        # 开始进行化妆                
        choice = input('你要选择哪个效果？1釉面 2亮面 3退出\n')
        
        # 如果选择了退出
        if choice == '3':
            break
        elif choice == '1' or choice == '2':  
            while True:
                image_url = input('请输入图片的地址:\n')
                print('你输入的地址为:'+image_url)
                if image_url.find('\\')>0:
                    print('图片地址格式有误!\n')
                elif os.path.exists(image_url):
                    break
                else:
                    print('图片地址不存在请重新输入!\n')
                
            iamge_name = image_url[image_url.rfind('/') + 1:image_url.rfind('.')]
            
            # 嘴部位置定位 
            im,temp_bgr,faces = mu.read_and_mark(image_url)
            mouth_top = faces[image_url][0].organs['mouth'].top
            mouth_bottom = faces[image_url][0].organs['mouth'].bottom
            mouth_left = faces[image_url][0].organs['mouth'].left
            mouth_right = faces[image_url][0].organs['mouth'].right    
            mouth_location[image_url] = [mouth_top, mouth_bottom, mouth_left, mouth_right]    
        
          
            # 定位嘴巴
            im = Image.open(image_url)   
            im = im.crop((mouth_left, mouth_top, mouth_right, mouth_bottom))
            mouth_im = im.resize((mouth_w, mouth_h))
            pix = mouth_im.load()
            
            mouth_matrix = np.zeros((mouth_h, mouth_w, chanel), dtype =np.float32)
            for h in range(mouth_h):
                for w in range(mouth_w):
                    for c in range(chanel):
                        mouth_matrix[h][w][c] = pix[w,h][c]              
            mouth_out = np.zeros((mouth_h, mouth_w, chanel), dtype =np.float32)
        
        
            # FCN网络预测
            if choice == '1':
                p = ai_makeup.predict_brightening(mouth_matrix, mouth_out)
                # p = brightening_sess.run(h_fc1, feed_dict={x: [mouth_matrix], y_: [mouth_out], keep_prob: 0.5})
                save_url = os.path.join(setting.s_pred_out_dir, '%s釉面.jpg' % iamge_name)

            elif choice == '2':
                # FCN网络预测
                p = ai_makeup.predict_whitening(mouth_matrix, mouth_out)
                # p = whitening_sess.run(h_fc1, feed_dict={x: [mouth_matrix], y_: [mouth_out], keep_prob: 0.5})
                save_url = os.path.join(setting.s_pred_out_dir, '%s亮面.jpg' % iamge_name)
                    
            print('计算已经完成！')
            
            # 嘴部位置定位
            mouth_top = mouth_location[image_url][0]
            mouth_bottom = mouth_location[image_url][1]
            mouth_left = mouth_location[image_url][2]
            mouth_right = mouth_location[image_url][3]
                
            # 图像的预测赋值
            mouth_matrix = np.zeros((mouth_h, mouth_w, chanel), dtype = np.float32)
            for h in range(mouth_h):
                for w in range(mouth_w): 
                    for c in range(chanel):
                        if p[0][h][w][c] > 255:
                            mouth_matrix[h,w,c] = 255.0
                        elif p[0][h][w][c] < 0:
                            mouth_matrix[h,w,c] = 0.0
                        else:
                            mouth_matrix[h,w,c] = p[0][h][w][c]
                            
                                               
            # 嘴巴进行尺度还原
            mouth_im = Image.fromarray(mouth_matrix.astype('uint8')) # 矩阵转图片对象     
            resolution = (mouth_right - mouth_left + 1, mouth_bottom - mouth_top + 1) # 原始嘴巴的分辨率
            
            mouth_resize_im = mouth_im.resize(resolution, Image.ANTIALIAS) # 还原嘴巴尺寸  
            mouth_resize_pix = mouth_resize_im.load()
                   
            print('mouth predict finish!')
          
            
            # 原始图片的处理
            raw_im = Image.open(image_url)
            raw_pix = raw_im.load()  
            raw_w, raw_h  = raw_im.size
            raw_matrix = np.zeros((raw_h, raw_w, chanel), dtype = np.float32)
            
            # 原始图片赋值
            for h in range(raw_h):
                for w in range(raw_w): 
                    for c in range(chanel):
                        raw_matrix[h, w, c] = raw_pix[w,h][c]
                        
          
            # 替换嘴唇图片
            for h in range(mouth_top, mouth_bottom +1):
                for w in range(mouth_left, mouth_right +1):   
                    raw_matrix[h,w] = mouth_resize_pix[w - int(mouth_left), h - int(mouth_top)]                                 
            print('face predict finish!') 
            
            new_im = Image.fromarray(raw_matrix.astype('uint8')) # 矩阵转为图对象
            if new_im.mode != 'RGB': # 图片不是RGB模型则转为RGB模式
                new_im = new_im.convert('RGB')
            new_im.save(save_url) # 保存验证集图片
            print('输出地址位于%s文件夹' % setting.s_pred_out_dir)
            print('\n')
        
    exit()
    