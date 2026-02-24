clc
clear all
close all


%%
clc
clear all

v5_01 = csvread('node15_vext0_05_stimulateonlyAbeta0_edgedist0.1_.csv');
v5_3 = csvread('node15_vext0_05_stimulateonlyAbeta0_edgedist3_.csv');

time = v5_01(:,1) ;

figure
plot(time , v5_01(:,2)  , 'g')
hold on
plot(time , v5_3(:,2)  , 'b')





%% dist = 0.1


Data = csvread('nodexg0changed_ALLExtraVoltages_stimulateonlyAbeta0_edgedist0.1_.csv');


time = Data(:,1) ;

[satr, soton] = size(Data) ;

Vext5 = Data(:,2) ;
Vext1 = Data(:,3) ;
Vext4 = Data(:,4) ;
Vext2 = Data(:,5) ;
Vext3 = Data(:,6) ;
boundary = Data(:,7)  ;

Vext6 = Data(:,8) ;
Vext7 = Data(:,9) ;
Vext8 = Data(:,10) ;
Vext9 = Data(:,11) ;



%
XG =  7.08e+04 *1e-8  ;   %S/micron2
xr = 95769.75706051444*1e6*1e-4 ;   %ohm/micron
Rpn0 = 429128.527578172*1e6*1e-4 ;  %ohm/micron
mycm=0.1    ;
mygm=0.001  ;
nl = 104 ;

xg0_MYSA  = (mygm/(nl*2))*1e-8 ;  %S/micron2
C_MYSA = (mycm/(nl*2))*1e-6*1e-8   ;     %Farad/micron2

V78 = (Vext7 - Vext8) ;
dvdt78 = gradient(V78(:)) ./ gradient(time(:));

I1 = (Vext2-Vext1)/xr ;    %mA.micron
I2 = (Vext1-Vext3)*XG ;     %mA/micron2
I3 = (Vext4-Vext1)/xr   ;    %mA.micron
I4 = (Vext5-Vext1)*XG       ;  %mA/micron2

IB = (Vext1 - boundary)*16.1*1e-8 ;    %mA/micron2

IG = Vext1*1e-9 *1e-8     ;  %mA/micron2

I6 = ((Vext6 - Vext5)/Rpn0)  ;  %mA.micron
I7 = ((Vext7 - Vext6)/Rpn0)  ;  %mA.micron
I8 = (Vext8 - Vext2)/xr  ;     %mA.micron
I9 = (Vext7 - Vext8)* xg0_MYSA  ;   %mA/micron2
I10 = C_MYSA * dvdt78  ;    %mA/micron2

I11 = (Vext9 - Vext5)/Rpn0  ;  %mA.micron

%

%%%% mA
IG = IG* 2.594156*pi ; 
I2 = I2* 2.594156*pi ;   %mA
I4 = I4* 2.594156*pi ;
IB = IB* 2.594156*pi ;

I9 = I9* 3*pi*8 ;
I10 = I10* 3*pi*8 ;


%%%%% mA
I1 = I1/0.5  ;
I3 = I3/0.5  ;
I6 = I6/0.5  ;
I11 = I11/0.5  ;

I7 = I7/1.5  ;
I8 = I8/1.5  ;


%%  dist = 0.5

 
d_05_Data = csvread('nodexg0changed_ALLExtraVoltages_stimulateonlyAbeta0_edgedist0.5_.csv');


time = d_05_Data(:,1) ;

[satr, soton] = size(d_05_Data) ;

d_05_Vext5 = d_05_Data(:,2) ;
d_05_Vext1 = d_05_Data(:,3) ;
d_05_Vext4 = d_05_Data(:,4) ;
d_05_Vext2 = d_05_Data(:,5) ;
d_05_Vext3 = d_05_Data(:,6) ;
d_05_boundary = d_05_Data(:,7)  ;

d_05_Vext6 = d_05_Data(:,8) ;
d_05_Vext7 = d_05_Data(:,9) ;
d_05_Vext8 = d_05_Data(:,10) ;
d_05_Vext9 = d_05_Data(:,11) ;



%
d_05_XG =   1.42e+04 *1e-8  ;   %S/micron2
d_05_xr = 18689.613196051898*1e6*1e-4 ;   %ohm/micron
d_05_Rpn0 = 429128.527578172*1e6*1e-4 ;  %ohm/micron
d_05_mycm=0.1    ;
d_05_mygm=0.001  ;
d_05_nl = 104 ;

d_05_xg0_MYSA  = (d_05_mygm/(d_05_nl*2))*1e-8 ;  %S/micron2
d_05_C_MYSA = (d_05_mycm/(d_05_nl*2))*1e-6*1e-8   ;     %Farad/micron2

d_05_V78 = (d_05_Vext7 - d_05_Vext8) ;
d_05_dvdt78 = gradient(d_05_V78(:)) ./ gradient(time(:));

d_05_I1 = (d_05_Vext2-d_05_Vext1)/d_05_xr ;    %mA.micron
d_05_I2 = (d_05_Vext1-d_05_Vext3)*d_05_XG ;     %mA/micron2
d_05_I3 = (d_05_Vext4-d_05_Vext1)/d_05_xr   ;    %mA.micron
d_05_I4 = (d_05_Vext5-d_05_Vext1)*d_05_XG       ;  %mA/micron2

d_05_IB = (d_05_Vext1 - d_05_boundary)*16.1*1e-8 ;    %mA/micron2

d_05_IG = d_05_Vext1*1e-9 *1e-8     ;  %mA/micron2

d_05_I6 = ((d_05_Vext6 - d_05_Vext5)/d_05_Rpn0)  ;  %mA.micron
d_05_I7 = ((d_05_Vext7 - d_05_Vext6)/d_05_Rpn0)  ;  %mA.micron
d_05_I8 = (d_05_Vext8 - d_05_Vext2)/d_05_xr  ;     %mA.micron
d_05_I9 = (d_05_Vext7 - d_05_Vext8)* d_05_xg0_MYSA  ;   %mA/micron2
d_05_I10 = d_05_C_MYSA * d_05_dvdt78  ;    %mA/micron2

d_05_I11 = (d_05_Vext9 - d_05_Vext5)/d_05_Rpn0  ;  %mA.micron

%

%%%% mA
d_05_IG = d_05_IG* 2.594156*pi ; 
d_05_I2 = d_05_I2* 2.594156*pi ;   %mA
d_05_I4 = d_05_I4* 2.594156*pi ;
d_05_IB = d_05_IB* 2.594156*pi ;

d_05_I9 = d_05_I9* 3*pi*8 ;
d_05_I10 = d_05_I10* 3*pi*8 ;


%%%%% mA
d_05_I1 = d_05_I1/0.5  ;
d_05_I3 = d_05_I3/0.5  ;
d_05_I6 = d_05_I6/0.5  ;
d_05_I11 = d_05_I11/0.5  ;

d_05_I7 = d_05_I7/1.5  ;
d_05_I8 = d_05_I8/1.5  ;


%%  dist = 1

 
d_1_Data = csvread('nodexg0changed_ALLExtraVoltages_stimulateonlyAbeta0_edgedist1_.csv');


time = d_1_Data(:,1) ;

[satr, soton] = size(d_1_Data) ;

d_1_Vext5 = d_1_Data(:,2) ;
d_1_Vext1 = d_1_Data(:,3) ;
d_1_Vext4 = d_1_Data(:,4) ;
d_1_Vext2 = d_1_Data(:,5) ;
d_1_Vext3 = d_1_Data(:,6) ;
d_1_boundary = d_1_Data(:,7)  ;

d_1_Vext6 = d_1_Data(:,8) ;
d_1_Vext7 = d_1_Data(:,9) ;
d_1_Vext8 = d_1_Data(:,10) ;
d_1_Vext9 = d_1_Data(:,11) ;



%
d_1_XG =   7.08e+03 *1e-8  ;   %S/micron2
d_1_xr = 9069.959345142834*1e6*1e-4 ;   %ohm/micron
d_1_Rpn0 = 429128.527578172*1e6*1e-4 ;  %ohm/micron
d_1_mycm=0.1    ;
d_1_mygm=0.001  ;
d_1_nl = 104 ;

d_1_xg0_MYSA  = (d_1_mygm/(d_1_nl*2))*1e-8 ;  %S/micron2
d_1_C_MYSA = (d_1_mycm/(d_1_nl*2))*1e-6*1e-8   ;     %Farad/micron2

d_1_V78 = (d_1_Vext7 - d_1_Vext8) ;
d_1_dvdt78 = gradient(d_1_V78(:)) ./ gradient(time(:));

d_1_I1 = (d_1_Vext2-d_1_Vext1)/d_1_xr  ;    %mA.micron
d_1_I2 = (d_1_Vext1-d_1_Vext3)*d_1_XG ;     %mA/micron2
d_1_I3 = (d_1_Vext4-d_1_Vext1)/d_1_xr   ;    %mA.micron
d_1_I4 = (d_1_Vext5-d_1_Vext1)*d_1_XG       ;  %mA/micron2

d_1_IB = (d_1_Vext1 - d_1_boundary)*16.1*1e-8 ;    %mA/micron2

d_1_IG = d_1_Vext1*1e-9 *1e-8     ;  %mA/micron2

d_1_I6 = ((d_1_Vext6 - d_1_Vext5)/d_1_Rpn0)  ;  %mA.micron
d_1_I7 = ((d_1_Vext7 - d_1_Vext6)/d_1_Rpn0)  ;  %mA.micron
d_1_I8 = (d_1_Vext8 - d_1_Vext2)/d_1_xr  ;     %mA.micron
d_1_I9 = (d_1_Vext7 - d_1_Vext8)* d_1_xg0_MYSA  ;   %mA/micron2
d_1_I10 = d_1_C_MYSA * d_1_dvdt78  ;    %mA/micron2

d_1_I11 = (d_1_Vext9 - d_1_Vext5)/d_1_Rpn0  ;  %mA.micron

%

%%%% mA
d_1_IG = d_1_IG* 2.594156*pi ; 
d_1_I2 = d_1_I2* 2.594156*pi ;   %mA
d_1_I4 = d_1_I4* 2.594156*pi ;
d_1_IB = d_1_IB* 2.594156*pi ;

d_1_I9 = d_1_I9* 3*pi*8 ;
d_1_I10 = d_1_I10* 3*pi*8 ;


%%%%% mA
d_1_I1 = d_1_I1/0.5  ;
d_1_I3 = d_1_I3/0.5  ;
d_1_I6 = d_1_I6/0.5  ;
d_1_I11 = d_1_I11/0.5  ;

d_1_I7 = d_1_I7/1.5  ;
d_1_I8 = d_1_I8/1.5  ;





%%  dist = 2

 
d_2_Data = csvread('nodexg0changed_ALLExtraVoltages_stimulateonlyAbeta0_edgedist2_.csv');


time = d_2_Data(:,1) ;

[satr, soton] = size(d_2_Data) ;

d_2_Vext5 = d_2_Data(:,2) ;
d_2_Vext1 = d_2_Data(:,3) ;
d_2_Vext4 = d_2_Data(:,4) ;
d_2_Vext2 = d_2_Data(:,5) ;
d_2_Vext3 = d_2_Data(:,6) ;
d_2_boundary = d_2_Data(:,7)  ;

d_2_Vext6 = d_2_Data(:,8) ;
d_2_Vext7 = d_2_Data(:,9) ;
d_2_Vext8 = d_2_Data(:,10) ;
d_2_Vext9 = d_2_Data(:,11) ;



%
d_2_XG =   3.54e+03 *1e-8  ;   %S/micron2
d_2_xr = 4283.03635742856*1e6*1e-4 ;   %ohm/micron
d_2_Rpn0 = 429128.527578172*1e6*1e-4 ;  %ohm/micron
d_2_mycm=0.1    ;
d_2_mygm=0.001  ;
d_2_nl = 104 ;

d_2_xg0_MYSA  = (d_2_mygm/(d_2_nl*2))*1e-8 ;  %S/micron2
d_2_C_MYSA = (d_2_mycm/(d_2_nl*2))*1e-6*1e-8   ;     %Farad/micron2

d_2_V78 = (d_2_Vext7 - d_2_Vext8) ;
d_2_dvdt78 = gradient(d_2_V78(:)) ./ gradient(time(:));

d_2_I1 = (d_2_Vext2-d_2_Vext1)/d_2_xr   ;    %mA.micron
d_2_I2 = (d_2_Vext1-d_2_Vext3)*d_2_XG ;     %mA/micron2
d_2_I3 = (d_2_Vext4-d_2_Vext1)/d_2_xr   ;    %mA.micron
d_2_I4 = (d_2_Vext5-d_2_Vext1)*d_2_XG       ;  %mA/micron2

d_2_IB = (d_2_Vext1 - d_2_boundary)*16.1*1e-8 ;    %mA/micron2

d_2_IG = d_2_Vext1*1e-9 *1e-8     ;  %mA/micron2

d_2_I6 = ((d_2_Vext6 - d_2_Vext5)/d_2_Rpn0)  ;  %mA.micron
d_2_I7 = ((d_2_Vext7 - d_2_Vext6)/d_2_Rpn0)  ;  %mA.micron
d_2_I8 = (d_2_Vext8 - d_2_Vext2)/d_2_xr  ;     %mA.micron
d_2_I9 = (d_2_Vext7 - d_2_Vext8)* d_2_xg0_MYSA  ;   %mA/micron2
d_2_I10 = d_2_C_MYSA * d_2_dvdt78  ;    %mA/micron2

d_2_I11 = (d_2_Vext9 - d_2_Vext5)/d_2_Rpn0  ;  %mA.micron

%

%%%% mA
d_2_IG = d_2_IG* 2.594156*pi ; 
d_2_I2 = d_2_I2* 2.594156*pi ;   %mA
d_2_I4 = d_2_I4* 2.594156*pi ;
d_2_IB = d_2_IB* 2.594156*pi ;

d_2_I9 = d_2_I9* 3*pi*8 ;
d_2_I10 = d_2_I10* 3*pi*8 ;


%%%%% mA
d_2_I1 = d_2_I1/0.5  ;
d_2_I3 = d_2_I3/0.5  ;
d_2_I6 = d_2_I6/0.5  ;
d_2_I11 = d_2_I11/0.5  ;

d_2_I7 = d_2_I7/1.5  ;
d_2_I8 = d_2_I8/1.5  ;







%%  dist = 3

 
d_3_Data = csvread('nodexg0changed_ALLExtraVoltages_stimulateonlyAbeta0_edgedist3_.csv');


time = d_3_Data(:,1) ;

[satr, soton] = size(d_3_Data) ;

d_3_Vext5 = d_3_Data(:,2) ;
d_3_Vext1 = d_3_Data(:,3) ;
d_3_Vext4 = d_3_Data(:,4) ;
d_3_Vext2 = d_3_Data(:,5) ;
d_3_Vext3 = d_3_Data(:,6) ;
d_3_boundary = d_3_Data(:,7)  ;

d_3_Vext6 = d_3_Data(:,8) ;
d_3_Vext7 = d_3_Data(:,9) ;
d_3_Vext8 = d_3_Data(:,10) ;
d_3_Vext9 = d_3_Data(:,11) ;



%
d_3_XG =   2.36e+03 *1e-8  ;   %S/micron2
d_3_xr = 2705.075594165407*1e6*1e-4  ;   %ohm/micron
d_3_Rpn0 = 429128.527578172*1e6*1e-4 ;  %ohm/micron
d_3_mycm=0.1    ;
d_3_mygm=0.001  ;
d_3_nl = 104 ;

d_3_xg0_MYSA  = (d_3_mygm/(d_3_nl*2))*1e-8 ;  %S/micron2
d_3_C_MYSA = (d_3_mycm/(d_3_nl*2))*1e-6*1e-8   ;     %Farad/micron2

d_3_V78 = (d_3_Vext7 - d_3_Vext8) ;
d_3_dvdt78 = gradient(d_3_V78(:)) ./ gradient(time(:));

d_3_I1 = (d_3_Vext2-d_3_Vext1)/d_3_xr  ;    %mA.micron
d_3_I2 = (d_3_Vext1-d_3_Vext3)*d_3_XG ;     %mA/micron2
d_3_I3 = (d_3_Vext4-d_3_Vext1)/d_3_xr   ;    %mA.micron
d_3_I4 = (d_3_Vext5-d_3_Vext1)*d_3_XG       ;  %mA/micron2

d_3_IB = (d_3_Vext1 - d_3_boundary)*16.1*1e-8 ;    %mA/micron2

d_3_IG = d_3_Vext1*1e-9 *1e-8     ;  %mA/micron2

d_3_I6 = ((d_3_Vext6 - d_3_Vext5)/d_3_Rpn0)  ;  %mA.micron
d_3_I7 = ((d_3_Vext7 - d_3_Vext6)/d_3_Rpn0)  ;  %mA.micron
d_3_I8 = (d_3_Vext8 - d_3_Vext2)/d_3_xr  ;     %mA.micron
d_3_I9 = (d_3_Vext7 - d_3_Vext8)* d_3_xg0_MYSA  ;   %mA/micron2
d_3_I10 = d_3_C_MYSA * d_3_dvdt78  ;    %mA/micron2

d_3_I11 = (d_3_Vext9 - d_3_Vext5)/d_3_Rpn0  ;  %mA.micron

%

%%%% mA
d_3_IG = d_3_IG* 2.594156*pi ; 
d_3_I2 = d_3_I2* 2.594156*pi ;   %mA
d_3_I4 = d_3_I4* 2.594156*pi ;
d_3_IB = d_3_IB* 2.594156*pi ;

d_3_I9 = d_3_I9* 3*pi*8 ;
d_3_I10 = d_3_I10* 3*pi*8 ;


%%%%% mA
d_3_I1 = d_3_I1/0.5  ;
d_3_I3 = d_3_I3/0.5  ;
d_3_I6 = d_3_I6/0.5  ;
d_3_I11 = d_3_I11/0.5  ;

d_3_I7 = d_3_I7/1.5  ;
d_3_I8 = d_3_I8/1.5  ;



%%


figure
plot(time, I1*1e6  , 'LineWidth' , 2) 
hold on
plot(time, d_05_I1*1e6  , 'LineWidth' , 2) 
hold on
plot(time, d_1_I1*1e6  , 'LineWidth' , 2) 
hold on
plot(time, d_2_I1*1e6  , 'LineWidth' , 2) 
hold on
plot(time, d_3_I1*1e6 , 'LineWidth' , 2) 

legend('0.1','0.5','1','2','3')

title('I1')
 xlim([0 1])

% figure
% plot(time, I3*1e6,  'LineWidth' , 2)
% hold on
% plot(time, d_3_I3*1e6,  'LineWidth' , 2)
% title('I3')
%  xlim([1.2 1.25])

% 
figure
plot(time, I6*1e6,  'LineWidth' , 2)
hold on
plot(time, d_05_I6*1e6,  'LineWidth' , 2)
hold on
plot(time, d_1_I6*1e6,  'LineWidth' , 2)
hold on
plot(time, d_2_I6*1e6,  'LineWidth' , 2)
hold on
plot(time, d_3_I6*1e6,  'LineWidth' , 2)
legend('0.1','0.5','1','2','3')
title('I6')
 xlim([0 1])




% figure
% plot(time, I11*1e6 , 'LineWidth' , 2)
% hold on
% plot(time, d_3_I11*1e6 , 'LineWidth' , 2)
% title('I11')
%  xlim([1.2 1.25])


% figure
% plot(time, I7*1e6,  'LineWidth' , 2)
% hold on
% plot(time, d_3_I7*1e6,  'LineWidth' , 2)
% title('I7')
% xlim([1.2 1.25])


% figure
% plot(time, I8*1e6,  'LineWidth' , 2)
% hold on
% plot(time, d_3_I8*1e6,  'LineWidth' , 2)
% title('I8')
%  xlim([1.2 1.25])


% 
% %%%%%%%
figure
plot(time, I2*1e6 ,  'LineWidth' , 2)
hold on
plot(time, d_05_I2*1e6 ,  'LineWidth' , 2)
hold on
plot(time, d_1_I2*1e6 ,  'LineWidth' , 2)
hold on
plot(time, d_2_I2*1e6 ,  'LineWidth' , 2)
hold on
plot(time, d_3_I2*1e6 ,  'LineWidth' , 2)
legend('0.1','0.5','1','2','3')
title('I2')
 xlim([0 1])

% 
% 
% 
% % 
figure
plot(time, -I4*1e6 ,  'LineWidth' , 2)
hold on
plot(time, -d_05_I4*1e6 , 'LineWidth' , 2)
hold on
plot(time, -d_1_I4*1e6 , 'LineWidth' , 2)
hold on
plot(time, -d_2_I4*1e6 , 'LineWidth' , 2)
hold on
plot(time, -d_3_I4*1e6 , 'LineWidth' , 2)
legend('0.1','0.5','1','2','3')
title('I4')
xlim([0 1])

% % 
% % 
% % 
% % 
% % 
figure
plot(time, IB*1e6  ,'LineWidth' , 2)
hold on
plot(time, d_05_IB*1e6  ,  'LineWidth' , 2)
hold on
plot(time, d_1_IB*1e6  ,  'LineWidth' , 2)
hold on
plot(time, d_2_IB*1e6  ,  'LineWidth' , 2)
hold on
plot(time, d_3_IB*1e6  ,  'LineWidth' , 2)
legend('0.1','0.5','1','2','3')
title('IB')
 xlim([0 1])

% 
% 
% 
% 
% 
% 
figure
plot(time, I9*1e6 , 'LineWidth' , 2)
hold on
plot(time, d_05_I9*1e6 , 'LineWidth' , 2)
hold on
plot(time, d_1_I9*1e6 , 'LineWidth' , 2)
hold on
plot(time, d_2_I9*1e6 , 'LineWidth' , 2)
hold on
plot(time, d_3_I9*1e6 , 'LineWidth' , 2)
legend('0.1','0.5','1','2','3')
title('I9')
 xlim([0 1])
% 
% 
% 
figure
plot(time, I10*1e6 , 'LineWidth' , 2)
hold on
plot(time, d_05_I10*1e6 , 'LineWidth' , 2)
hold on
plot(time, d_1_I10*1e6 , 'LineWidth' , 2)
hold on
plot(time, d_2_I10*1e6 , 'LineWidth' , 2)
hold on
plot(time, d_3_I10*1e6 , 'LineWidth' , 2)
legend('0.1','0.5','1','2','3')
title('I10')

 xlim([0 1])
 
 
%  
%  %%
% close all
%  
% %%  
% 
% 
% v01 = csvread('v_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');
% v05 = csvread('v_Abeta0_stimulateonlyAbeta0_edgedist0.5_.csv');
% v1 = csvread('v_Abeta0_stimulateonlyAbeta0_edgedist1_.csv');
% v2 = csvread('v_Abeta0_stimulateonlyAbeta0_edgedist2_.csv');
% v3 = csvread('v_Abeta0_stimulateonlyAbeta0_edgedist3_.csv');
% 
% 
% 
% 
%  figure
%  
% plot(time, v01(:,17)+Vext5 , 'LineWidth' , 2)
% hold on
% plot(time, v05(:,17)+d_05_Vext5 , 'LineWidth' , 2)
% hold on
% plot(time, v1(:,17)+d_1_Vext5 , 'LineWidth' , 2)
% hold on
% plot(time, v2(:,17)+d_2_Vext5 , 'LineWidth' , 2)
% hold on
% plot(time, v3(:,17)+d_3_Vext5 , 'LineWidth' , 2)
% hold on
% 
% title('node intracellular voltage')
% legend('0.1','0.5','1','2','3')
%  
%%

 figure
 
plot(time, Vext5 , 'LineWidth' , 2)
hold on
plot(time, d_05_Vext5 , 'LineWidth' , 2)
hold on
plot(time, d_1_Vext5 , 'LineWidth' , 2)
hold on
plot(time, d_2_Vext5 , 'LineWidth' , 2)
hold on
plot(time, d_3_Vext5 , 'LineWidth' , 2)
hold on
xlim([0 1])
title('V5')
legend('0.1','0.5','1','2','3')

%%

%  figure
%  
% plot(time, Vext1 , ':' , 'LineWidth' , 2)
% hold on
% plot(time, d_05_Vext1 , ':' ,'LineWidth' , 2)
% hold on
% plot(time, d_1_Vext1 , ':' ,'LineWidth' , 2)
% hold on
% plot(time, d_2_Vext1 , ':' ,'LineWidth' , 2)
% hold on
% plot(time, d_3_Vext1 , ':' ,'LineWidth' , 2)
% hold on
% 
% title('V1')
% legend('0.1','0.5','1','2','3')

%%
% 
%  figure
%  
% plot(time, Vext6 , 'LineWidth' , 2)
% hold on
% plot(time, d_05_Vext6 , 'LineWidth' , 2)
% hold on
% plot(time, d_1_Vext6 , 'LineWidth' , 2)
% hold on
% plot(time, d_2_Vext6 , 'LineWidth' , 2)
% hold on
% plot(time, d_3_Vext6 , 'LineWidth' , 2)
% hold on
% 
% title('V6')
% legend('0.1','0.5','1','2','3')
% 
% %%
% 
%  figure
%  
% plot(time, Vext9 , 'LineWidth' , 2)
% hold on
% plot(time, d_05_Vext9 , 'LineWidth' , 2)
% hold on
% plot(time, d_1_Vext9 , 'LineWidth' , 2)
% hold on
% plot(time, d_2_Vext9 , 'LineWidth' , 2)
% hold on
% plot(time, d_3_Vext9 , 'LineWidth' , 2)
% hold on
% 
% title('V9')
% legend('0.1','0.5','1','2','3')
% 
% 
%  %%
% 
%  figure
%  
% plot(time, Vext2 , 'LineWidth' , 2)
% hold on
% plot(time, d_05_Vext2 , 'LineWidth' , 2)
% hold on
% plot(time, d_1_Vext2 , 'LineWidth' , 2)
% hold on
% plot(time, d_2_Vext2 , 'LineWidth' , 2)
% hold on
% plot(time, d_3_Vext2 , 'LineWidth' , 2)
% hold on
% 
% title('V2')
% legend('0.1','0.5','1','2','3')
% 
% 
% %%
% 
%  figure
%  
% plot(time, Vext4 , 'LineWidth' , 2)
% hold on
% plot(time, d_05_Vext4 , 'LineWidth' , 2)
% hold on
% plot(time, d_1_Vext4 , 'LineWidth' , 2)
% hold on
% plot(time, d_2_Vext4 , 'LineWidth' , 2)
% hold on
% plot(time, d_3_Vext4 , 'LineWidth' , 2)
% hold on
% 
% title('V4')
% legend('0.1','0.5','1','2','3')
% 
% 
% %%
% 
%  figure
%  
% plot(time, Vext7 , 'LineWidth' , 2)
% hold on
% plot(time, d_05_Vext7 , 'LineWidth' , 2)
% hold on
% plot(time, d_1_Vext7 , 'LineWidth' , 2)
% hold on
% plot(time, d_2_Vext7 , 'LineWidth' , 2)
% hold on
% plot(time, d_3_Vext7 , 'LineWidth' , 2)
% hold on
% 
% title('V7')
% legend('0.1','0.5','1','2','3')
% 
% 
% %%
% 
%  figure
%  
% plot(time, Vext8 , 'LineWidth' , 2)
% hold on
% plot(time, d_05_Vext8 , 'LineWidth' , 2)
% hold on
% plot(time, d_1_Vext8 , 'LineWidth' , 2)
% hold on
% plot(time, d_2_Vext8 , 'LineWidth' , 2)
% hold on
% plot(time, d_3_Vext8 , 'LineWidth' , 2)
% hold on
% 
% title('V8')
% legend('0.1','0.5','1','2','3')
% 
% %%
% close all
%   figure
%  
% plot(time, Vext3 , ':' ,'LineWidth' , 2)
% hold on
% plot(time, d_05_Vext3 ,':' , 'LineWidth' , 2)
% hold on
% plot(time, d_1_Vext3 ,':' , 'LineWidth' , 2)
% hold on
% plot(time, d_2_Vext3 , ':' ,'LineWidth' , 2)
% hold on
% plot(time, d_3_Vext3 ,':' ,'LineWidth' , 2)
% hold on
% 
% title('V3 aligned')
% legend('0.1','0.5','1','2','3')
% 



