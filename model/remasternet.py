import torch
import torch.nn as nn
import torch.nn.functional as F

class TempConv( nn.Module ):
   def __init__(self, in_planes, out_planes, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1) ):
      super(TempConv, self).__init__()
      self.conv3d  = nn.Conv3d( in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding )
      self.bn      = nn.BatchNorm3d( out_planes )
   def forward(self, x):
      return F.elu( self.bn( self.conv3d( x ) ), inplace=False )

class Upsample( nn.Module ):
   def __init__(self, in_planes, out_planes, scale_factor=(1,2,2)):
      super(Upsample, self).__init__()
      self.scale_factor = scale_factor
      self.conv3d = nn.Conv3d( in_planes, out_planes, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1) )
      self.bn   = nn.BatchNorm3d( out_planes )
   def forward(self, x):
      return F.elu( self.bn( self.conv3d( F.interpolate(x, scale_factor=self.scale_factor, mode='trilinear', align_corners=False) ) ), inplace=False )

class UpsampleConcat( nn.Module ):
   def __init__(self, in_planes_up, in_planes_flat, out_planes):
      super(UpsampleConcat, self).__init__()
      self.conv3d = TempConv( in_planes_up + in_planes_flat, out_planes, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1) )
   def forward(self, x1, x2):
      x1 = F.interpolate(x1, scale_factor=(1,2,2), mode='trilinear', align_corners=False)
      x = torch.cat([x1, x2], dim=1)
      return self.conv3d( x )

class SourceReferenceAttention(nn.Module):
    """
    Source-Reference Attention Layer
    """
    def __init__(self, in_planes_s, in_planes_r):
        """
        Parameters
        ----------
            in_planes_s: int
                Number of input source feature vector channels.
            in_planes_r: int
                Number of input reference feature vector channels.
        """
        super(SourceReferenceAttention,self).__init__()
        self.query_conv = nn.Conv3d( in_channels=in_planes_s,
                out_channels=in_planes_s//8, kernel_size=1 )
        self.key_conv   = nn.Conv3d( in_channels=in_planes_r,
                out_channels=in_planes_r//8, kernel_size=1 )
        self.value_conv = nn.Conv3d( in_channels=in_planes_r,
                out_channels=in_planes_r,    kernel_size=1 )
        self.gamma      = nn.Parameter(torch.zeros(1))
        self.softmax    = nn.Softmax(dim=-1)
    def forward(self, source, reference):
        """
        Parameters
        ----------
            source : torch.Tensor
                Source feature maps (B x Cs x Ts x Hs x Ws)
            reference : torch.Tensor
                Reference feature maps (B x Cr x Tr x Hr x Wr )
         Returns :
            torch.Tensor
                Source-reference attention value added to the input source features
            torch.Tensor
                Attention map (B x Ns x Nt) (Ns=Ts*Hs*Ws, Nr=Tr*Hr*Wr)
        """
        s_batchsize, sC, sT, sH, sW = source.size()
        r_batchsize, rC, rT, rH, rW = reference.size()
        proj_query = self.query_conv(source).view(s_batchsize,-1,sT*sH*sW).permute(0,2,1)
        proj_key   = self.key_conv(reference).view(r_batchsize,-1,rT*rW*rH)
        energy     = torch.bmm( proj_query, proj_key )
        attention  = self.softmax(energy)
        proj_value = self.value_conv(reference).view(r_batchsize,-1,rT*rH*rW)
        out    = torch.bmm(proj_value,attention.permute(0,2,1) )
        out    = out.view(s_batchsize, sC, sT, sH, sW)
        out    = self.gamma*out + source
        return out, attention


class NetworkR( nn.Module ):
   def __init__(self):
      super(NetworkR, self).__init__()

      self.layers = nn.Sequential(
         nn.ReplicationPad3d((1,1,1,1,1,1)),
         TempConv(   1,  64, kernel_size=(3,3,3), stride=(1,2,2), padding=(0,0,0) ),
         TempConv(  64, 128, kernel_size=(3,3,3), padding=(1,1,1) ),
         TempConv( 128, 128, kernel_size=(3,3,3), padding=(1,1,1) ),
         TempConv( 128, 256, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1) ),
         TempConv( 256, 256, kernel_size=(3,3,3), padding=(1,1,1) ),
         TempConv( 256, 256, kernel_size=(3,3,3), padding=(1,1,1) ),
         TempConv( 256, 256, kernel_size=(3,3,3), padding=(1,1,1) ),
         TempConv( 256, 256, kernel_size=(3,3,3), padding=(1,1,1) ),
         Upsample( 256, 128 ),
         TempConv( 128,  64, kernel_size=(3,3,3), padding=(1,1,1) ),
         TempConv(  64,  64, kernel_size=(3,3,3), padding=(1,1,1) ),
         Upsample( 64, 16 ),
         nn.Conv3d( 16, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1) )
       )
   def forward(self, x):
      return (x + torch.tanh( self.layers( x.clone()-0.4462414 ) )).clamp(0,1)

class NetworkC( nn.Module ):
   def __init__(self):
      super(NetworkC, self).__init__()

      self.down1 = nn.Sequential(
         nn.ReplicationPad3d((1,1,1,1,0,0)),
         TempConv(   1,  64, stride=(1,2,2), padding=(0,0,0) ),
         TempConv(  64, 128 ),
         TempConv( 128, 128 ),
         TempConv( 128, 256, stride=(1,2,2) ),
         TempConv( 256, 256 ),
         TempConv( 256, 256 ),
         TempConv( 256, 512, stride=(1,2,2) ),
         TempConv( 512, 512 ),
         TempConv( 512, 512 )                  
      )
      self.flat = nn.Sequential(
         TempConv( 512, 512 ),
         TempConv( 512, 512 )
      )
      self.down2 = nn.Sequential(
         TempConv( 512, 512, stride=(1,2,2) ),
         TempConv( 512, 512 ),
      )
      self.stattn1 = SourceReferenceAttention( 512, 512 ) # Source-Reference Attention
      self.stattn2 = SourceReferenceAttention( 512, 512 ) # Source-Reference Attention
      self.selfattn1 = SourceReferenceAttention( 512, 512 ) # Self Attention
      self.conv1 = TempConv( 512, 512 )
      self.up1 = UpsampleConcat( 512, 512, 512 ) # 1/8
      self.selfattn2 = SourceReferenceAttention( 512, 512 ) # Self Attention
      self.conv2 = TempConv( 512, 256, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1) )
      self.up2 = nn.Sequential(
         Upsample( 256, 128 ), # 1/4
         TempConv( 128, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1) )
      )      
      self.up3 = nn.Sequential(
         Upsample( 64, 32 ), # 1/2
         TempConv( 32, 16, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1) )
      )
      self.up4 = nn.Sequential(
         Upsample( 16, 8 ), # 1/1
         nn.Conv3d( 8, 2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1) )
      )
      self.reffeatnet1 = nn.Sequential(
         TempConv(   3,  64, stride=(1,2,2) ),
         TempConv(  64, 128 ),
         TempConv( 128, 128 ),
         TempConv( 128, 256, stride=(1,2,2) ),
         TempConv( 256, 256 ),
         TempConv( 256, 256 ),
         TempConv( 256, 512, stride=(1,2,2) ),
         TempConv( 512, 512 ),
         TempConv( 512, 512 ),
      )
      self.reffeatnet2 = nn.Sequential(
         TempConv( 512, 512, stride=(1,2,2) ),
         TempConv( 512, 512 ),
         TempConv( 512, 512 ),
      )

   def forward(self, x, x_refs=None):
      x1 = self.down1( x - 0.4462414 )

      if x_refs is not None:
         x_refs = x_refs.transpose(2,1).contiguous() # [B,T,C,H,W] --> [B,C,T,H,W]
         reffeat = self.reffeatnet1( x_refs-0.48 )
         x1, _ = self.stattn1( x1, reffeat )

      x2 = self.flat( x1 )
      out = self.down2( x1 )

      if x_refs is not None:
         reffeat2 = self.reffeatnet2( reffeat )
         out, _ = self.stattn2( out, reffeat2 )
      
      out = self.conv1( out )
      out, _ = self.selfattn1( out, out )
      out = self.up1( out, x2 )
      out, _ = self.selfattn2( out, out )
      out = self.conv2( out )
      out = self.up2( out )
      out = self.up3( out )
      out = self.up4( out )

      return torch.sigmoid( out )