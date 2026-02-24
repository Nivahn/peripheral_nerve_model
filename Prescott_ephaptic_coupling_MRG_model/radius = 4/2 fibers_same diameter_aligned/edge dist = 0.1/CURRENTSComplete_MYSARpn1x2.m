clc
clear all
%close all

%

Surface_area_node = 2.59415600000000*pi*1e-8 ;  %cm2
Surface_area_MYSA = 8*3*pi*1e-8 ;  %cm2
Surface_area_FLUT = 8*8.05888*pi*1e-8 ;  %cm2
Surface_area_STIN = 8*19.641280*pi*1e-8 ;  %cm2



%

Left_Trans_Leak = csvread('MYSARpn1x2_TransmembraneCurrent_Leak_Leftside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');

Left_Trans_Cap = csvread('MYSARpn1x2_TransmembraneCurrent_Capacitive_Leftside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');


Right_Trans_Leak = csvread('MYSARpn1x2_TransmembraneCurrent_Leak_Rightside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');

Right_Trans_Cap = csvread('MYSARpn1x2_TransmembraneCurrent_Capacitive_Rightside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');




[soton satr] = size(Left_Trans_Leak) ;
time = Left_Trans_Leak(:,1)  ;

%  changing the unit from mA/cm2  to nA


for i=2:22
    
    Left_Trans_Leak(:,i) = Left_Trans_Leak(:,i)*Surface_area_STIN ;  % mA
    Left_Trans_Leak(:,i) = Left_Trans_Leak(:,i)*1e6  ; % nA
end

for i=23:27
    
    Left_Trans_Leak(:,i) = Left_Trans_Leak(:,i)*Surface_area_FLUT ;  % mA
    Left_Trans_Leak(:,i) = Left_Trans_Leak(:,i)*1e6  ; % nA
end


for i=28
    
    Left_Trans_Leak(:,i) = Left_Trans_Leak(:,i)*Surface_area_MYSA ;  % mA
    Left_Trans_Leak(:,i) = Left_Trans_Leak(:,i)*1e6  ; % nA
end


%%%%%%%%%%%%%%%

for i=2:22
    
    Left_Trans_Cap(:,i) = Left_Trans_Cap(:,i)*Surface_area_STIN ;  % mA
    Left_Trans_Cap(:,i) = Left_Trans_Cap(:,i)*1e6  ; % nA
end

for i=23:27
    
    Left_Trans_Cap(:,i) = Left_Trans_Cap(:,i)*Surface_area_FLUT  ; % mA
    Left_Trans_Cap(:,i) = Left_Trans_Cap(:,i)*1e6  ; % nA
end


for i=28
    
    Left_Trans_Cap(:,i) = Left_Trans_Cap(:,i)*Surface_area_MYSA  ; % mA
    Left_Trans_Cap(:,i) = Left_Trans_Cap(:,i)*1e6  ; % nA
end







for i=2:22
    
    Right_Trans_Leak(:,i) = Right_Trans_Leak(:,i)*Surface_area_STIN ;  % mA
    Right_Trans_Leak(:,i) = Right_Trans_Leak(:,i)*1e6  ; % nA
end

for i=23:27
    
    Right_Trans_Leak(:,i) = Right_Trans_Leak(:,i)*Surface_area_FLUT ;  % mA
    Right_Trans_Leak(:,i) = Right_Trans_Leak(:,i)*1e6  ; % nA
end


for i=28
    
    Right_Trans_Leak(:,i) = Right_Trans_Leak(:,i)*Surface_area_MYSA ;  % mA
    Right_Trans_Leak(:,i) = Right_Trans_Leak(:,i)*1e6  ; % nA
end


%%%%%%%%%%%%%%%

for i=2:22
    
    Right_Trans_Cap(:,i) = Right_Trans_Cap(:,i)*Surface_area_STIN ;  % mA
    Right_Trans_Cap(:,i) = Right_Trans_Cap(:,i)*1e6  ; % nA
end

for i=23:27
    
    Right_Trans_Cap(:,i) = Right_Trans_Cap(:,i)*Surface_area_FLUT  ; % mA
    Right_Trans_Cap(:,i) = Right_Trans_Cap(:,i)*1e6  ; % nA
end


for i=28
    
    Right_Trans_Cap(:,i) = Right_Trans_Cap(:,i)*Surface_area_MYSA  ; % mA
    Right_Trans_Cap(:,i) = Right_Trans_Cap(:,i)*1e6  ; % nA
end



%
startColor = [1, 1, 1];
xx = startColor ;

figure
for i=2:satr
    if i<25
        m=0.008 ;
        xx = xx - m  ;
    else
        m=0.2  ;   
        xx = xx - m  ;
    end
    plot(time , Left_Trans_Leak(:,i) ,'--' ,'color', xx , 'LineWidth' , 2 ) ;

    hold on
  
end

xlabel('time(ms)')
ylabel('transmembrane-il (nA)')
title('Left side')
xlim([0.95 4])


% for i=2:satr
%    
%     plot(time , Right_Trans_Leak(:,i) , 'LineWidth' , 2)
%     hold on
%     
%     
% end
% xlabel('time(ms)')
% ylabel('transmembrane_il (nA)')
% title('Right side')






%%


xx = startColor ;


figure
for i=2:satr
    
    if i<25
        m=0.008 ;
        xx = xx - m  ;
    else
        m=0.2  ;   
        xx = xx - m  ;
    end
   
    plot(time , Left_Trans_Cap(:,i) ,'--', 'color', xx , 'LineWidth' , 2)
    hold on
    
    
end
xlabel('time(ms)')
ylabel('icap (nA)')
title('Left side')
xlim([0.95 4])


% for i=2:satr
%    
%     plot(time , Right_Trans_Cap(:,i) , 'LineWidth' , 2)
%     hold on
%     
%     
% end
% xlabel('time(ms)')
% ylabel('icap (nA)')
% title('Right side')








%%   vext0

Rpn0 = 429128.527578172*1e6*1e-4 ;  %ohm/micron   node and MYSA
Rpn2 = 108066.18906880693*1e6*1e-4 ;  %ohm/micron  FLUT
Rpx = 108066.18906880693*1e6*1e-4 ;  %ohm/micron    STIN





Left_vext0 = csvread('MYSARpn1x2_vext0_Leftside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');



%

j=1 ;
for i=2:42
    
    I_vext0_left(:,j) = (Left_vext0(:,i) - Left_vext0(:,i+1))/Rpx  ;   %mA.micron
    I_vext0_left(:,j) = I_vext0_left(:,j)/9.8205   ;   %mA
    I_vext0_left(:,j) = I_vext0_left(:,j)*1e6  ;       % nA

    j=j+1 ; 
    
end



for i=43:52
    
    I_vext0_left(:,j) = (Left_vext0(:,i) - Left_vext0(:,i+1))/Rpn2  ;   %mA.micron
    I_vext0_left(:,j) = I_vext0_left(:,j)/4.0294   ;   %mA
    I_vext0_left(:,j) = I_vext0_left(:,j)*1e6  ;       % nA

    j=j+1 ; 
    
end



for i=53:54
    
    I_vext0_left(:,j) = (Left_vext0(:,i) - Left_vext0(:,i+1))/Rpn0  ;   %mA.micron
    I_vext0_left(:,j) = I_vext0_left(:,j)/1.5   ;   %mA
    I_vext0_left(:,j) = I_vext0_left(:,j)*1e6  ;       % nA

    j=j+1 ; 
    
end


for i=55
    
    I_vext0_left(:,j) = (Left_vext0(:,i) - Left_vext0(:,i+1))/Rpn0  ;   %mA.micron
    I_vext0_left(:,j) = I_vext0_left(:,j)/0.5   ;   %mA
    I_vext0_left(:,j) = I_vext0_left(:,j)*1e6  ;       % nA

    j=j+1 ; 
    
end

startColor = [1, 1, 1];
xx = startColor ;
figure
for i=2:54
   
    
    if i<48
        m=0.005 ;
        xx = xx - m  ;
    else
        m=0.1  ;   
        xx = xx - m  ;
    end
    plot(time , I_vext0_left(:,i) ,'--' , 'Color' , xx ,'LineWidth' , 2)
    hold on
    
    
end
xlabel('time(ms)')
ylabel('periaxonal current (nA)')
title('Left side')
xlim([0.95 4])




%%








Right_vext0 = csvread('MYSARpn1x2_vext0_Rightside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');







j=1 ;
for i=2:42
    
    I_vext0_right(:,j) = (Right_vext0(:,i) - Right_vext0(:,i+1))/Rpx  ;   %mA.micron
    I_vext0_right(:,j) = I_vext0_right(:,j)/9.8205   ;   %mA
    I_vext0_right(:,j) = I_vext0_right(:,j)*1e6  ;       % nA

    j=j+1 ; 
    
end



for i=43:52
    
    I_vext0_right(:,j) = (Right_vext0(:,i) - Right_vext0(:,i+1))/Rpn2  ;   %mA.micron
    I_vext0_right(:,j) = I_vext0_right(:,j)/4.0294   ;   %mA
    I_vext0_right(:,j) = I_vext0_right(:,j)*1e6  ;       % nA

    j=j+1 ; 
    
end



for i=53:54
    
    I_vext0_right(:,j) = (Right_vext0(:,i) - Right_vext0(:,i+1))/Rpn0  ;   %mA.micron
    I_vext0_right(:,j) = I_vext0_right(:,j)/1.5   ;   %mA
    I_vext0_right(:,j) = I_vext0_right(:,j)*1e6  ;       % nA

    j=j+1 ; 
    
end


for i=55
    
    I_vext0_right(:,j) = (Right_vext0(:,i) - Right_vext0(:,i+1))/Rpn0  ;   %mA.micron
    I_vext0_right(:,j) = I_vext0_right(:,j)/0.5   ;   %mA
    I_vext0_right(:,j) = I_vext0_right(:,j)*1e6  ;       % nA

    j=j+1 ; 
    
end



% figure
% for i=2:54
%    
%     plot(time , I_vext0_right(:,i) , '--' , 'LineWidth' , 2)
%     hold on
%     
%     
% end
% xlabel('time(ms)')
% ylabel('periaxonal current (nA)')
% title('Right side')



%%  vext1

xr = 95769.75706051444*1e6*1e-4 ;   %ohm/micron



Left_vext1 = csvread('MYSARpn1x2_vext1_Leftside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');






%

j=1 ;
for i=2:42
    
    I_vext1_left(:,j) = (Left_vext1(:,i) - Left_vext1(:,i+1))/xr   ;   %mA.micron
    I_vext1_left(:,j) = I_vext1_left(:,j)/9.8205   ;   %mA
    I_vext1_left(:,j) = I_vext1_left(:,j)*1e6  ;       % nA

    j=j+1 ; 
    
end



for i=43:52
    
    I_vext1_left(:,j) = (Left_vext1(:,i) - Left_vext1(:,i+1))/xr  ;   %mA.micron
    I_vext1_left(:,j) = I_vext1_left(:,j)/4.0294   ;   %mA
    I_vext1_left(:,j) = I_vext1_left(:,j)*1e6  ;       % nA

    j=j+1 ; 
    
end



for i=53:54
    
    I_vext1_left(:,j) = (Left_vext1(:,i) - Left_vext1(:,i+1))/xr  ;   %mA.micron
    I_vext1_left(:,j) = I_vext1_left(:,j)/1.5   ;   %mA
    I_vext1_left(:,j) = I_vext1_left(:,j)*1e6  ;       % nA

    j=j+1 ; 
    
end


for i=55
    
    I_vext1_left(:,j) = (Left_vext1(:,i) - Left_vext1(:,i+1))/xr  ;   %mA.micron
    I_vext1_left(:,j) = I_vext1_left(:,j)/0.5   ;   %mA
    I_vext1_left(:,j) = I_vext1_left(:,j)*1e6  ;       % nA

    j=j+1 ; 
    
end



% figure

% startColor = [1, 1, 1];
% xx = startColor ;
% 
% 
% for i=2:54
%     
%     
%     
%     if i<30
%         m=0.005 ;
%         xx = xx - m  ;
%     else
%         m=0.1  ;   
%         xx = xx - m  ;
%     end
%    
%     plot(time , I_vext1_left(:,i) ,'--' ,  'LineWidth' , 2)
%     hold on
%     
%     
% end
xlabel('time(ms)')
ylabel('Longitudinal current (nA)')
title('Left side')
xlim([0.95 4])




%%








Right_vext1 = csvread('MYSARpn1x2_vext1_Rightside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');







j=1 ;
for i=2:42
    
    I_vext1_right(:,j) = (Right_vext1(:,i) - Right_vext1(:,i+1))/xr   ;   %mA.micron
    I_vext1_right(:,j) = I_vext1_right(:,j)/9.8205   ;   %mA
    I_vext1_right(:,j) = I_vext1_right(:,j)*1e6  ;       % nA

    j=j+1 ; 
    
end



for i=43:52
    
    I_vext1_right(:,j) = (Right_vext1(:,i) - Right_vext1(:,i+1))/xr  ;   %mA.micron
    I_vext1_right(:,j) = I_vext1_right(:,j)/4.0294   ;   %mA
    I_vext1_right(:,j) = I_vext1_right(:,j)*1e6  ;       % nA

    j=j+1 ; 
    
end



for i=53:54
    
    I_vext1_right(:,j) = (Right_vext1(:,i) - Right_vext1(:,i+1))/xr  ;   %mA.micron
    I_vext1_right(:,j) = I_vext1_right(:,j)/1.5   ;   %mA
    I_vext1_right(:,j) = I_vext1_right(:,j)*1e6  ;       % nA

    j=j+1 ; 
    
end


for i=55
    
    I_vext1_right(:,j) = (Right_vext1(:,i) - Right_vext1(:,i+1))/xr  ;   %mA.micron
    I_vext1_right(:,j) = I_vext1_right(:,j)/0.5   ;   %mA
    I_vext1_right(:,j) = I_vext1_right(:,j)*1e6  ;       % nA

    j=j+1 ; 
    
end


startColor = [1, 1, 1];
xx = startColor ;
% figure
% 
% for i=1:54
%     
%     
%     if i<48
%         m=0.005 ;
%         xx = xx - m  ;
%     else
%         m=0.1  ;   
%         xx = xx - m  ;
%     end
%    
%     plot(time , I_vext1_right(:,i)  , 'LineWidth' , 2)
%     hold on
%     
%     
% end
xlabel('time(ms)')
ylabel('Longitudinal current (nA)')
title('Right side')
xlim([0.95 4])



%%


Surface_area_node = 2.59415600000000*pi ;  %micron2
Surface_area_MYSA = 8*3*pi ;  
Surface_area_FLUT = 8*8.05888*pi ;  
Surface_area_STIN = 8*19.641280*pi ;  


%%%%%%%%  I4

XG =  7.08e+04 *1e-8  ;   %S/micron2

IT = (Left_vext0(:,56)-Right_vext1(:,56))*XG   ;


IT = IT * Surface_area_node*1e6  ; 

figure
plot(time , IT  , '--' ,  'LineWidth' , 2)


xlabel('time(ms)')
ylabel('IT (nA)')
xlim([0.95 4])

%% to the neaby fiber

N =  csvread('MYSARpn1x2_vext1_Abeta1_stimulateonlyAbeta0_edgedist0.1_.csv');


I_nearby = (Right_vext1(:,56)-N(:,2))*XG   ;
I_nearby = I_nearby * Surface_area_node*1e6  ; 

figure
plot(time , I_nearby  , '--', 'LineWidth' , 2)


xlabel('time(ms)')
ylabel('I-nearby (nA)')
xlim([0.95 4])

%% to boundary


B =  csvread('MYSARpn1x2_Boundary_stimulateonlyAbeta0_edgedist0.1_.csv');


I_boundary = (Right_vext1(:,56)-B(:,2))*16.1*1e-8   ;
I_boundary = I_boundary * Surface_area_node*1e6  ; 


figure
plot(time , I_boundary , '--' , 'LineWidth' , 2)


xlabel('time(ms)')
ylabel('I-boundary (nA)')
xlim([0.95 4])

%%  node icap and Na

Surface_area_node = 2.59415600000000*pi*1e-8 ;  %cm2


node_icapp = csvread('MYSARpn1x2_icap_Node15Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');  %mA/cm2


node_icap = node_icapp(:,2)*Surface_area_node*1e6  ;


figure
plot(time , node_icap , 'LineWidth' , 2)
xlabel('time(ms)')
ylabel('node,icap (nA)')



node_inaa = csvread('MYSARpn1x2_ina_Node15Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');  %mA/cm2

node_ina = node_inaa(:,2)*Surface_area_node*1e6  ;


figure
plot(time , node_ina , 'LineWidth' , 2)
xlabel('time(ms)')
ylabel('node,ina (nA)')
xlim([0.95 4])


%%  

mycm=0.1    ;
mygm=0.001  ;
nl = 104 ;

xg0_MYSA  = (mygm/(nl*2))*1e-8 ;  %S/micron2
C_MYSA = (mycm/(nl*2))*1e-6*1e-8   ;     %Farad/micron2


k=1 ;
for i=2:2:42
    
    
Left_myelin_leak(:,k) = (Left_vext0(:,i)-Left_vext1(:,i))* xg0_MYSA    ;
Left_myelin_leak(:,k) = Left_myelin_leak(:,k)*Surface_area_STIN*1e6  ;


clear C
C = (Left_vext0(:,i)-Left_vext1(:,i))  ;
dvdt = gradient(C(:)) ./ gradient(time(:));

Left_myelin_cap(:,k) = dvdt*C_MYSA ;
Left_myelin_cap(:,k) = Left_myelin_cap(:,k) *Surface_area_STIN*1e6  ;



k=k+1 ;

end


for i=44:2:52
    
Left_myelin_leak(:,k) = (Left_vext0(:,i)-Left_vext1(:,i))* xg0_MYSA    ;
Left_myelin_leak(:,k) = Left_myelin_leak(:,k)*Surface_area_FLUT*1e6  ;

clear C
C = (Left_vext0(:,i)-Left_vext1(:,i))  ;
dvdt = gradient(C(:)) ./ gradient(time(:));

Left_myelin_cap(:,k) = dvdt*C_MYSA ;
Left_myelin_cap(:,k) = Left_myelin_cap(:,k) *Surface_area_FLUT*1e6  ;


k=k+1 ;

end


for i=54
    
Left_myelin_leak(:,k) = (Left_vext0(:,i)-Left_vext1(:,i))* xg0_MYSA    ;
Left_myelin_leak(:,k) = Left_myelin_leak(:,k)*Surface_area_MYSA*1e6  ;

clear C
C = (Left_vext0(:,i)-Left_vext1(:,i))  ;
dvdt = gradient(C(:)) ./ gradient(time(:));

Left_myelin_cap(:,k) = dvdt*C_MYSA ;
Left_myelin_cap(:,k) = Left_myelin_cap(:,k) *Surface_area_MYSA*1e6  ;





k=k+1 ;

end

startColor = [1, 1, 1];
xx = startColor ;
figure
for i=1:27
    
    if i<20
        m=0.005 ;
        xx = xx - m  ;
    else
        m=0.1  ;   
        xx = xx - m  ;
    end
   
    plot(time , Left_myelin_leak(:,i) , '--' , 'LineWidth' , 2)
    hold on
    
    
end
xlabel('time(ms)')
ylabel('myelin-leak (nA)')
title('left')


startColor = [1, 1, 1];
xx = startColor ;
figure
for i=1:27
    
    if i<20
        m=0.005 ;
        xx = xx - m  ;
    else
        m=0.1  ;   
        xx = xx - m  ;
    end
   
    plot(time , Left_myelin_cap(:,i) , '--' , 'LineWidth' , 2)
    hold on
    
    
end

xlabel('time(ms)')
ylabel('myelin-cap (nA)')
title('left')
xlim([0.95 4])
%% 


mycm=0.1    ;
mygm=0.001  ;
nl = 104 ;

xg0_MYSA  = (mygm/(nl*2))*1e-8 ;  %S/micron2
C_MYSA = (mycm/(nl*2))*1e-6*1e-8   ;     %Farad/micron2


k=1 ;
for i=2:2:42
    
Right_myelin_leak(:,k) = (Right_vext0(:,i)-Right_vext1(:,i))* xg0_MYSA    ;
Right_myelin_leak(:,k) = Right_myelin_leak(:,k)*Surface_area_STIN*1e6  ;

clear C
C = (Right_vext0(:,i)-Right_vext1(:,i))  ;
dvdt = gradient(C(:)) ./ gradient(time(:));

right_myelin_cap(:,k) = dvdt*C_MYSA ;
right_myelin_cap(:,k) = right_myelin_cap(:,k) *Surface_area_STIN*1e6  ;




k=k+1 ;

end


for i=44:2:52
    
Right_myelin_leak(:,k) = (Right_vext0(:,i)-Right_vext1(:,i))* xg0_MYSA    ;
Right_myelin_leak(:,k) = Right_myelin_leak(:,k)*Surface_area_FLUT*1e6  ;

clear C
C = (Right_vext0(:,i)-Right_vext1(:,i))  ;
dvdt = gradient(C(:)) ./ gradient(time(:));

right_myelin_cap(:,k) = dvdt*C_MYSA  ;
right_myelin_cap(:,k) = right_myelin_cap(:,k) *Surface_area_FLUT*1e6  ;




k=k+1 ;

end


for i=54
    
Right_myelin_leak(:,k) = (Right_vext0(:,i)-Right_vext1(:,i))* xg0_MYSA    ;
Right_myelin_leak(:,k) = Right_myelin_leak(:,k)*Surface_area_MYSA*1e6  ;

clear C
C = (Right_vext0(:,i)-Right_vext1(:,i))  ;
dvdt = gradient(C(:)) ./ gradient(time(:));
 
right_myelin_cap(:,k) = dvdt*C_MYSA  ;
right_myelin_cap(:,k) = right_myelin_cap(:,k) *Surface_area_MYSA*1e6  ;




k=k+1 ;

end

startColor = [1, 1, 1];
xx = startColor ;
figure
for i=1:27
   
    
    if i<20
        m=0.005 ;
        xx = xx - m  ;
    else
        m=0.1  ;   
        xx = xx - m  ;
    end
    plot(time , Right_myelin_leak(:,i) , '--' , 'LineWidth' , 2)
    hold on
    
    
end

xlabel('time(ms)')
ylabel('myelin-leak (nA)')
title('right')
xlim([0.95 4])


%%
startColor = [1, 1, 1];
xx = startColor ;
figure
for i=20:27
    
    
    if i<20
        m=0.005 ;
        xx = xx - m  ;
    else
        m=0.1  ;   
        xx = xx - m  ;
    end
   
    plot(time , right_myelin_cap(:,i) , '--' , 'Color' ,xx, 'LineWidth' , 2)
    hold on
    
    
end

xlabel('time(ms)')
ylabel('myelin-cap (nA)')
title('right')
xlim([0.95 4])











