clc
clear all
close all

%%


Data = csvread('MYSAandFLUTExtraVoltages_vext0_stimulateonlyAbeta0_edgedist0.1_.csv');


time = Data(:,1) ;


V1 = Data(:,2) ;
V2 = Data(:,3) ;
V3 = Data(:,4) ;
V4 = Data(:,5) ;
V5 = Data(:,6) ;
V6 = Data(:,7)  ;
V7 = Data(:,8) ;
V8 = Data(:,9) ;
V9 = Data(:,10) ;
V10 = Data(:,11) ;
V11 = Data(:,12) ;
V12 = Data(:,13) ;
V13 = Data(:,14) ;


Rpn0 = 429128.527578172*1e6*1e-4 ;  %ohm/micron
Rpn2 = 108066.18906880693*1e6*1e-4 ;  %ohm/micron  
Rpx = 108066.18906880693*1e6*1e-4 ;  %ohm/micron  


I1 = (V1-V2)/Rpn0  ;    %mA.micron
I2 = (V2-V3)/Rpn0  ;    %mA.micron
I3 = (V3-V4)/Rpn2  ;    %mA.micron
I4 = (V4-V5)/Rpn2  ;    %mA.micron
I5 = (V5-V6)/Rpn2  ;    %mA.micron
I6 = (V6-V7)/Rpn2  ;    %mA.micron
I7 = (V7-V8)/Rpn2  ;    %mA.micron
I8 = (V8-V9)/Rpn2  ;    %mA.micron
I9 = (V9-V10)/Rpn2  ;    %mA.micron
I10 = (V10-V11)/Rpn2  ;    %mA.micron
I11 = (V11-V12)/Rpn2  ;    %mA.micron
I12 = (V12-V13)/Rpn2  ;    %mA.micron






%%%%% mA
I1 = I1/1.5  ;
I2 = I2/1.5  ;



L = 8.0589/2  ;


I3 = I3/L  ;
I4 = I4/L  ;
I5 = I5/L  ;
I6 = I6/L  ;
I7 = I7/L  ;
I8 = I8/L  ;
I9 = I9/L  ;
I10 = I10/L  ;
I11 = I11/L  ;
I12 = I12/L  ;



%%

Data2 = csvread('STINExtraVoltages_vext0_stimulateonlyAbeta0_edgedist0.1_.csv');

u=2 ; 
V14 = Data2(:,u) ;
u=u+1 ;
V15 = Data2(:,u) ;
u=u+1 ;
V16 = Data2(:,u) ;
u=u+1 ;
V17 = Data2(:,u) ;
u=u+1 ;
V18 = Data2(:,u) ;
u=u+1 ;
V19 = Data2(:,u) ;
u=u+1 ;
V20 = Data2(:,u) ;
u=u+1 ;
V21 = Data2(:,u) ;
u=u+1 ;
V22 = Data2(:,u) ;
u=u+1 ;
V23 = Data2(:,u) ;
u=u+1 ;
V24 = Data2(:,u) ;
u=u+1 ;
V25 = Data2(:,u) ;
u=u+1 ;
V26 = Data2(:,u) ;
u=u+1 ;
V27 = Data2(:,u) ;
u=u+1 ;
V28 = Data2(:,u) ;
u=u+1 ;
V29 = Data2(:,u) ;
u=u+1 ;
V30 = Data2(:,u) ;
u=u+1 ;
V31 = Data2(:,u) ;
u=u+1 ;
V32 = Data2(:,u) ;
u=u+1 ;
V33 = Data2(:,u) ;
u=u+1 ;
V34 = Data2(:,u) ;
u=u+1 ;
V35 = Data2(:,u) ;
u=u+1 ;
V36 = Data2(:,u) ;
u=u+1 ;
V37 = Data2(:,u) ;
u=u+1 ;
V38 = Data2(:,u) ;
u=u+1 ;
V39 = Data2(:,u) ;
u=u+1 ;
V40 = Data2(:,u) ;
u=u+1 ;
V41 = Data2(:,u) ;
u=u+1 ;
V42 = Data2(:,u) ;
u=u+1 ;
V43 = Data2(:,u) ;
u=u+1 ;
V44 = Data2(:,u) ;
u=u+1 ;
V45 = Data2(:,u) ;
u=u+1 ;
V46 = Data2(:,u) ;
u=u+1 ;
V47 = Data2(:,u) ;
u=u+1 ;
V48 = Data2(:,u) ;
u=u+1 ;
V49 = Data2(:,u) ;
u=u+1 ;
V50 = Data2(:,u) ;
u=u+1 ;
V51 = Data2(:,u) ;
u=u+1 ;
V52 = Data2(:,u) ;
u=u+1 ;
V53 = Data2(:,u) ;
u=u+1 ;
V54 = Data2(:,u) ;




I13 = (V13-V14)/Rpn2  ;    %mA.micron
I14 = (V14-V15)/Rpn2  ;    %mA.micron
I15 = (V15-V16)/Rpn2  ;    %mA.micron
I16 = (V16-V17)/Rpn2  ;    %mA.micron
I17 = (V17-V18)/Rpn2  ;    %mA.micron
I18 = (V18-V19)/Rpn2  ;    %mA.micron
I19 = (V19-V20)/Rpn2  ;    %mA.micron
I20 = (V20-V21)/Rpn2  ;    %mA.micron
I21 = (V21-V22)/Rpn2  ;    %mA.micron
I22 = (V22-V23)/Rpn2  ;    %mA.micron
I23 = (V23-V24)/Rpn2  ;    %mA.micron
I24 = (V24-V25)/Rpn2  ;    %mA.micron
I25 = (V25-V26)/Rpn2  ;    %mA.micron
I26 = (V26-V27)/Rpn2  ;    %mA.micron
I27 = (V27-V28)/Rpn2  ;    %mA.micron
I28 = (V28-V29)/Rpn2  ;    %mA.micron
I29 = (V29-V30)/Rpn2  ;    %mA.micron
I30 = (V30-V31)/Rpn2  ;    %mA.micron
I31 = (V31-V32)/Rpn2  ;    %mA.micron
I32 = (V32-V33)/Rpn2  ;    %mA.micron
I33 = (V33-V34)/Rpn2  ;    %mA.micron
I34 = (V34-V35)/Rpn2  ;    %mA.micron
I35 = (V35-V36)/Rpn2  ;    %mA.micron
I36 = (V36-V37)/Rpn2  ;    %mA.micron
I37 = (V37-V38)/Rpn2  ;    %mA.micron
I38 = (V38-V39)/Rpn2  ;    %mA.micron
I39 = (V39-V40)/Rpn2  ;    %mA.micron
I40 = (V40-V41)/Rpn2  ;    %mA.micron
I41 = (V41-V42)/Rpn2  ;    %mA.micron
I42 = (V42-V43)/Rpn2  ;    %mA.micron
I43 = (V43-V44)/Rpn2  ;    %mA.micron
I44 = (V44-V45)/Rpn2  ;    %mA.micron
I45 = (V45-V46)/Rpn2  ;    %mA.micron
I46 = (V46-V47)/Rpn2  ;    %mA.micron
I47 = (V47-V48)/Rpn2  ;    %mA.micron
I48 = (V48-V49)/Rpn2  ;    %mA.micron
I49 = (V49-V50)/Rpn2  ;    %mA.micron
I50 = (V50-V51)/Rpn2  ;    %mA.micron
I51 = (V51-V52)/Rpn2  ;    %mA.micron
I52 = (V52-V53)/Rpn2  ;    %mA.micron
I53 = (V53-V54)/Rpn2  ;    %mA.micron




L2 = 19.641/2 ;



I13 = I13 /L2 ;
I14 = I14 /L2 ;
I15 = I15 /L2 ;
I16 = I16 /L2 ;
I17 = I17 /L2 ;
I18 = I18 /L2 ;
I19 = I19 /L2 ;
I20 = I20 /L2 ;
I21 = I21 /L2 ;
I22 = I22 /L2 ;
I23 = I23 /L2 ;
I24 = I24 /L2 ;
I25 = I25 /L2 ;
I26 = I26 /L2 ;
I27 = I27 /L2 ;
I28 = I28 /L2 ;
I29 = I29 /L2 ;
I30 = I30 /L2 ;
I31 = I31 /L2 ;
I32 = I32 /L2 ;
I33 = I33 /L2 ;
I34 = I34 /L2 ;
I35 = I35 /L2 ;
I36 = I36 /L2 ;
I37 = I37 /L2 ;
I38 = I38 /L2 ;
I39 = I39 /L2 ;
I40 = I40 /L2 ;
I41 = I41 /L2 ;
I42 = I42 /L2 ;
I43 = I43 /L2 ;
I44 = I44 /L2 ;
I45 = I45 /L2 ;
I46 = I46 /L2 ;
I47 = I47 /L2 ;
I48 = I48 /L2 ;
I49 = I49 /L2 ;
I50 = I50 /L2 ;
I51 = I51 /L2 ;
I52 = I52 /L2 ;
I53 = I53 /L2 ;



%%












 


% all currents
figure



plot(time, -I1*1e6  , 'LineWidth' , 2)   %nano amper
hold on
% 
plot(time, -I2*1e6 , 'LineWidth' , 2)
hold on
plot(time, -I3*1e6   , 'LineWidth' , 2)   %nano amper
hold on
plot(time, -I4*1e6  , 'LineWidth' , 2)   %nano amper
hold on
plot(time, -I5*1e6  , 'LineWidth' , 2)   %nano amper
hold on
plot(time, -I6*1e6   , 'LineWidth' , 2)   %nano amper
hold on
plot(time, -I7*1e6  , 'LineWidth' , 2)   %nano amper
hold on
plot(time, -I8*1e6   , 'LineWidth' , 2)   %nano amper
hold on
plot(time, -I9*1e6  , 'LineWidth' , 2)   %nano amper
hold on
plot(time, -I10*1e6   , 'LineWidth' , 2)   %nano amper
hold on
plot(time, -I11*1e6   , 'LineWidth' , 2)   %nano amper
hold on
plot(time, -I12*1e6   , 'LineWidth' , 2)   %nano amper
hold on



hold on
plot(time, -I13*1e6   , 'LineWidth' , 2)   %nano amper
hold on

plot(time, -I14*1e6   , 'LineWidth' , 2)   %nano amper
hold on
hold on
plot(time, -I15*1e6   , 'LineWidth' , 2)   %nano amper
hold on
hold on
plot(time, -I16*1e6   , 'LineWidth' , 2)   %nano amper
hold on
hold on
plot(time, -I17*1e6   , 'LineWidth' , 2)   %nano amper
hold on
hold on
plot(time, -I18*1e6   , 'LineWidth' , 2)   %nano amper
hold on
hold on
plot(time, -I19*1e6   , 'LineWidth' , 2)   %nano amper
hold on
hold on
plot(time, -I20*1e6   , 'LineWidth' , 2)   %nano amper
hold on
% hold on
% plot(time, -I21*1e6   , 'LineWidth' , 2)   %nano amper
% hold on
% hold on
% plot(time, -I22*1e6   , 'LineWidth' , 2)   %nano amper
% hold on
% hold on
% plot(time, -I23*1e6   , 'LineWidth' , 2)   %nano amper
% hold on
% hold on
% plot(time, -I24*1e6   , 'LineWidth' , 2)   %nano amper
% hold on
% hold on
% plot(time, -I25*1e6   , 'LineWidth' , 2)   %nano amper
% hold on
% hold on
% plot(time, -I26*1e6   , 'LineWidth' , 2)   %nano amper
% hold on
% hold on
% plot(time, -I27*1e6   , 'LineWidth' , 2)   %nano amper
% hold on
% plot(time, -I28*1e6   , 'LineWidth' , 2)   %nano amper
% hold on
% plot(time, -I29*1e6   , 'LineWidth' , 2)   %nano amper
% hold on
plot(time, -I30*1e6   , 'LineWidth' , 2)   %nano amper
hold on

plot(time, -I40*1e6   , 'LineWidth' , 2)   %nano amper
hold on
plot(time, -I50*1e6   , 'LineWidth' , 2)   %nano amper
hold on








