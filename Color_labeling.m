clear ; close all; clc

load matrix-3_dim.mat MB_img2
load idx1.mat m

idx=m;

%assign same value of every feature in a same cluster
x= idx==0;
aa=x*255;

y= idx==1;
bb=y*255;

z= idx==2;
cc=z*255;



%assigning colour to each cluster
%differentiate far ground ,Back Ground and noise.
MB_img(:,:,1)=aa;
MB_img(:,:,2)=bb;
MB_img(:,:,3)=cc;
figure(2)
imshow(MB_img)

