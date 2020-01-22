"""
   Copyright (C) <2019> <Satoshi Iizuka and Edgar Simo-Serra>

   This work is licensed under the Creative Commons
   Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy
   of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or
   send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

   Satoshi Iizuka, University of Tsukuba
   iizuka@aoni.waseda.jp, http://iizuka.cs.tsukuba.ac.jp/index_eng.html
   Edgar Simo-Serra, Waseda University
   ess@waseda.jp, https://esslab.jp/~ess/
"""

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import argparse
import subprocess
import utils

parser = argparse.ArgumentParser(description='Remastering')
parser.add_argument('--input',   type=str,   default='none', help='Input video')
parser.add_argument('--reference_dir',  type=str, default='none', help='Path to the reference image directory')
parser.add_argument('--disable_colorization', action='store_true', default=False, help='Remaster without colorization')
parser.add_argument('--gpu',       action='store_true', default=False, help='Use GPU')
parser.add_argument('--mindim',     type=int,   default='320',    help='Length of minimum image edges')
opt = parser.parse_args()

device = torch.device('cuda:0' if opt.gpu else 'cpu')

# Load remaster network
modelR = __import__( 'model.remasternet', fromlist=['NetworkR'] ).NetworkR()
state_dict = torch.load( 'model/remasternet.pth.tar' )
modelR.load_state_dict( state_dict['modelR'] )
modelR = modelR.to(device)
modelR.eval()
if not opt.disable_colorization:
   modelC = __import__( 'model.remasternet', fromlist=['NetworkC'] ).NetworkC()
   modelC.load_state_dict( state_dict['modelC'] )
   modelC = modelC.to(device)
   modelC.eval()

print('Processing %s...'%os.path.basename(opt.input))

outputdir = 'tmp/'
outputdir_in = outputdir+'input/'
os.makedirs( outputdir_in, exist_ok=True )
outputdir_out = outputdir+'output/'
os.makedirs( outputdir_out, exist_ok=True )

# Prepare reference images
if not opt.disable_colorization:
   if opt.reference_dir!='none':
      import glob
      ext_list = ['png','jpg','bmp']
      reference_files = []
      for ext in ext_list:
         reference_files += glob.glob( opt.reference_dir+'/*.'+ext, recursive=True )
      aspect_mean = 0
      minedge_dim = 256
      refs = []
      for v in reference_files:
         refimg = Image.open( v ).convert('RGB')
         w, h = refimg.size
         aspect_mean += w/h
         refs.append( refimg )
      aspect_mean /= len(reference_files)
      target_w = int(256*aspect_mean) if aspect_mean>1 else 256
      target_h = 256 if aspect_mean>=1 else int(256/aspect_mean)
      refimgs = torch.FloatTensor(len(reference_files), 3, target_h, target_w)
      for i, v in enumerate(refs):
         refimg = utils.addMergin( v, target_w=target_w, target_h=target_h )
         refimgs[i] = transforms.ToTensor()( refimg )
      refimgs = refimgs.view(1, refimgs.size(0), refimgs.size(1), refimgs.size(2), refimgs.size(3)).to( device )

# Load video
cap = cv2.VideoCapture( opt.input )
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # cv2.CAP_PROP_FRAME_COUNT: 7
v_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
v_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
minwh = min(v_w,v_h)
scale = 1
if minwh != opt.mindim:
   scale = opt.mindim / minwh
t_w = round(v_w*scale/16.)*16
t_h = round(v_h*scale/16.)*16
fps = cap.get(cv2.CAP_PROP_FPS)
pbar = tqdm(total=nframes)
block = 5

# Process 
with torch.no_grad():
   it = 0
   while True:
      frame_pos = it*block
      if frame_pos >= nframes:
         break
      cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
      if block >= nframes-frame_pos:
         proc_g = nframes-frame_pos
      else:
         proc_g = block

      input = None
      gtC = None
      for i in range(proc_g):
         index = frame_pos + i
         _, frame = cap.read()
         frame = cv2.resize(frame, (t_w, t_h))
         nchannels = frame.shape[2]
         if nchannels == 1 or not opt.disable_colorization:
            frame_l = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(outputdir_in+'%07d.png'%index, frame_l)
            frame_l = torch.from_numpy(frame_l).view( frame_l.shape[0], frame_l.shape[1], 1 )
            frame_l = frame_l.permute(2, 0, 1).float() # HWC to CHW
            frame_l /= 255.
            frame_l = frame_l.view(1, frame_l.size(0), 1, frame_l.size(1), frame_l.size(2))
         elif nchannels == 3:
            cv2.imwrite(outputdir_in+'%07d.png'%index, frame)
            frame = frame[:,:,::-1] ## BGR -> RGB
            frame_l, frame_ab = utils.convertRGB2LABTensor( frame )
            frame_l = frame_l.view(1, frame_l.size(0), 1, frame_l.size(1), frame_l.size(2))
            frame_ab = frame_ab.view(1, frame_ab.size(0), 1, frame_ab.size(1), frame_ab.size(2))

         input = frame_l if i==0 else torch.cat( (input, frame_l), 2 )
         if nchannels==3 and opt.disable_colorization:
            gtC = frame_ab if i==0 else torch.cat( (gtC, frame_ab), 2 )
      
      input = input.to( device )

      # Perform restoration
      output_l = modelR( input ) # [B, C, T, H, W]

      # Save restoration output without colorization when using the option [--disable_colorization]
      if opt.disable_colorization:
         for i in range( proc_g ):
            index = frame_pos + i
            if nchannels==3:
               out_l = output_l.detach()[0,:,i].cpu()
               out_ab = gtC[0,:,i].cpu()
               out = torch.cat((out_l, out_ab),dim=0).detach().numpy().transpose((1, 2, 0))
               out = Image.fromarray( np.uint8( utils.convertLAB2RGB( out )*255 ) )
               out.save( outputdir_out+'%07d.png'%(index) )
            else:
               save_image( output_l.detach()[0,:,i], outputdir_out+'%07d.png'%(index), nrow=1 )
      # Perform colorization
      else:
         if opt.reference_dir=='none':
            output_ab = modelC( output_l )
         else:
            output_ab = modelC( output_l, refimgs )
         output_l = output_l.detach().cpu()
         output_ab = output_ab.detach().cpu()
         
         # Save output frames of restoration with colorization
         for i in range( proc_g ):
            index = frame_pos + i
            out_l = output_l[0,:,i,:,:]
            out_c = output_ab[0,:,i,:,:]
            output = torch.cat((out_l, out_c), dim=0).numpy().transpose((1, 2, 0))
            output = Image.fromarray( np.uint8( utils.convertLAB2RGB( output )*255 ) )
            output.save( outputdir_out+'%07d.png'%index )

      it = it + 1
      pbar.update(proc_g)
   
   # Save result videos
   outfile = opt.input.split('/')[-1].split('.')[0]
   cmd = 'ffmpeg -y -r %d -i %s%%07d.png -vcodec libx264 -pix_fmt yuv420p -r %d %s_in.mp4' % (fps, outputdir_in, fps, outfile )
   subprocess.call( cmd, shell=True )
   cmd = 'ffmpeg -y -r %d -i %s%%07d.png -vcodec libx264 -pix_fmt yuv420p -r %d %s_out.mp4' % (fps, outputdir_out, fps, outfile )
   subprocess.call( cmd, shell=True )
   cmd = 'ffmpeg -y -i %s_in.mp4 -vf "[in] pad=2.01*iw:ih [left];movie=%s_out.mp4[right];[left][right] overlay=main_w/2:0,scale=2*iw/2:2*ih/2[out]" %s_comp.mp4' % ( outfile, outfile, outfile )
   subprocess.call( cmd, shell=True )

   import shutil
   shutil.rmtree(outputdir)
   cap.release()
   pbar.close()