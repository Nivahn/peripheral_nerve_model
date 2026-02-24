clc
clear all
close all

%% node15 transmembrane current

clc
clear all

Surface_area_node = 2.59415600000000*pi*1e-8 ;  %cm2

node15_imembrane_05 = csvread('xg0changed_node15_imembrane_stimulateonlyAbeta0_edgedist0.1_.csv');  %(mA/cm2)

time = node15_imembrane_05(:,1)  ;

node15_imembrane_05(:,2) = node15_imembrane_05(:,2)*Surface_area_node*1e6 ; %nA





figure
plot(time , -node15_imembrane_05(:,2)  , 'g' , 'LineWidth' , 1.5)
hold on


%%%%%%%%%%%%%%%%%%%

MYSARpn1x2_node15_imembrane_05 = csvread('MYSARpn1x2_xg0changed_node15_imembrane_stimulateonlyAbeta0_edgedist0.1_.csv');

time = MYSARpn1x2_node15_imembrane_05(:,1)  ;

MYSARpn1x2_node15_imembrane_05(:,2) = MYSARpn1x2_node15_imembrane_05(:,2)*Surface_area_node*1e6 ;


plot(time , -MYSARpn1x2_node15_imembrane_05(:,2)  , 'b' , 'LineWidth' , 1.5)
hold on

xlim([0 0.1])

xlabel('time (ms)')
ylabel('Transmembrane imembrane (nA)')





%% transmembrane voltage
clc
clear all

node15_v_05 = csvread('node15_V_05_stimulateonlyAbeta0_edgedist0.1_.csv');

time = node15_v_05(:,1)  ;

figure
plot(time , node15_v_05(:,2)  , 'g' , 'LineWidth' , 1.5)
hold on




MYSARpn1x2_node15_v_05 = csvread('MYSARpn1x2_node15_V_05_stimulateonlyAbeta0_edgedist0.1_.csv');

plot(time , MYSARpn1x2_node15_v_05(:,2)  , 'b' , 'LineWidth' , 1.5)
hold on


xlim([0 0.12])

xlabel('time (ms)')
ylabel('Transmembrane V (nA)')








%% Iin  (right)
clc
clear all

Right_vext0 = csvread('vext0_Rightside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');

node15_v_05 = csvread('node15_V_05_stimulateonlyAbeta0_edgedist0.1_.csv');

node15_v_1 = csvread('node15_v_1_stimulateonlyAbeta0_edgedist0.1_.csv');

time = node15_v_1(:,1)  ;

Iin = ( (node15_v_05(:,2)+Right_vext0(:,56) ) - (node15_v_1(:,2) + Right_vext0(:,55)  )) / (0.066219513*1e6)   ;   % r=0.066219513 Mohm (I used function ri(x) in NEURON to find it)   mV/ohm = mA
Iin = Iin*1e6  ; %nA

figure
plot(time , Iin  , 'g' , 'LineWidth' , 1.5)
hold on

%%%%%%%%%%%%%%%%%%%%%%%%%%%% MYSA Rpn1x2


MYSARpn1x2_Right_vext0 = csvread('MYSARpn1x2_vext0_Rightside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');

MYSARpn1x2_node15_v_05 = csvread('MYSARpn1x2_node15_V_05_stimulateonlyAbeta0_edgedist0.1_.csv');

MYSARpn1x2_node15_v_1 = csvread('MYSARpn1x2_node15_v_1_stimulateonlyAbeta0_edgedist0.1_.csv');

time = MYSARpn1x2_node15_v_1(:,1)  ;

MYSARpn1x2_Iin = ( (MYSARpn1x2_node15_v_05(:,2)+MYSARpn1x2_Right_vext0(:,56) ) - (MYSARpn1x2_node15_v_1(:,2) + MYSARpn1x2_Right_vext0(:,55)  )) / (0.066219513*1e6)   ;    % r=0.066219513 ohm   mV/ohm = mA
MYSARpn1x2_Iin = MYSARpn1x2_Iin*1e6  ; %nA


plot(time , MYSARpn1x2_Iin  , 'b' , 'LineWidth' , 1.5)
hold on
xlim([0 0.1])
title('Iin (rightside) (nA)')
xlabel('time (ms)')
ylabel('Current (nA)')


%% Iin  (left)
clc
clear all

Left_vext0 = csvread('vext0_Leftside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');

node15_v_05 = csvread('node15_V_05_stimulateonlyAbeta0_edgedist0.1_.csv');

node15_v_0 = csvread('node15_v_0_stimulateonlyAbeta0_edgedist0.1_.csv');

time = node15_v_0(:,1)  ;

Iin = ( (node15_v_05(:,2)+Left_vext0(:,56) ) - (node15_v_0(:,2) + Left_vext0(:,55)  )) / (0.066219513*1e6)   ;   % r=0.066219513 Mohm   mV/ohm = mA
Iin = Iin*1e6  ; %nA

figure
plot(time , Iin  , 'g' , 'LineWidth' , 1.5)
hold on

%%%%%%%%%%%%%%%%%%%%%%%%%%%% MYSA Rpn1x2


MYSARpn1x2_Left_vext0 = csvread('MYSARpn1x2_vext0_Leftside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');

MYSARpn1x2_node15_v_05 = csvread('MYSARpn1x2_node15_V_05_stimulateonlyAbeta0_edgedist0.1_.csv');

MYSARpn1x2_node15_v_0 = csvread('MYSARpn1x2_node15_v_0_stimulateonlyAbeta0_edgedist0.1_.csv');

time = MYSARpn1x2_node15_v_0(:,1)  ;

MYSARpn1x2_Iin = ( (MYSARpn1x2_node15_v_05(:,2)+MYSARpn1x2_Left_vext0(:,56) ) - (MYSARpn1x2_node15_v_0(:,2) + MYSARpn1x2_Left_vext0(:,55)  )) / (0.066219513*1e6)   ;    % r=0.066219513 ohm   mV/ohm = mA
MYSARpn1x2_Iin = MYSARpn1x2_Iin*1e6  ; %nA


plot(time , MYSARpn1x2_Iin  , 'b' , 'LineWidth' , 1.5)
hold on
xlim([0 0.12])
title('Iin (leftside) (nA)')
xlabel('time (ms)')
ylabel('Current (nA)')



%% MYSA  leak current (rightside)
clc
clear all

Right_Trans_Leak = csvread('TransmembraneCurrent_Leak_Rightside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');

time = Right_Trans_Leak(:,1) ;

Surface_area_MYSA = 8*3*pi*1e-8 ;  %cm2

MYSA30_trans_leak = Right_Trans_Leak(:,28)   ;  % mA/cm2
MYSA30_trans_leak = MYSA30_trans_leak * Surface_area_MYSA *1e6  ;   %nA

figure
plot(time , MYSA30_trans_leak  , 'g' , 'LineWidth' , 1.5)
hold on

%%%%%%%%%%%%%%%%%%%%%%% MYSA Rpn1 x2

MYSARpn1x2_Right_Trans_Leak = csvread('MYSARpn1x2_TransmembraneCurrent_Leak_Rightside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');

time = MYSARpn1x2_Right_Trans_Leak(:,1) ;

Surface_area_MYSA = 8*3*pi*1e-8 ;  %cm2

MYSARpn1x2_MYSA30_trans_leak = MYSARpn1x2_Right_Trans_Leak(:,28)   ;  % mA/cm2
MYSARpn1x2_MYSA30_trans_leak = MYSARpn1x2_MYSA30_trans_leak * Surface_area_MYSA *1e6  ;   %nA

plot(time , MYSARpn1x2_MYSA30_trans_leak  , 'b' , 'LineWidth' , 1.5)
hold on

xlim([0 0.12])
title('Paranode Ileak (rightside) (nA)')
xlabel('time (ms)')
ylabel('Current (nA)')

%% MYSA  leak current (leftside)
clc
clear all

Left_Trans_Leak = csvread('TransmembraneCurrent_Leak_Leftside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');

time = Left_Trans_Leak(:,1) ;

Surface_area_MYSA = 8*3*pi*1e-8 ;  %cm2

MYSA29_trans_leak = Left_Trans_Leak(:,28)   ;  % mA/cm2
MYSA29_trans_leak = MYSA29_trans_leak * Surface_area_MYSA *1e6  ;   %nA

figure
plot(time , MYSA29_trans_leak  , 'g' , 'LineWidth' , 1.5)
hold on

%%%%%%%%%%%%%%%%%%%%%%% MYSA Rpn1 x2

MYSARpn1x2_Left_Trans_Leak = csvread('MYSARpn1x2_TransmembraneCurrent_Leak_Leftside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');

time = MYSARpn1x2_Left_Trans_Leak(:,1) ;

Surface_area_MYSA = 8*3*pi*1e-8 ;  %cm2

MYSARpn1x2_MYSA29_trans_leak = MYSARpn1x2_Left_Trans_Leak(:,28)   ;  % mA/cm2
MYSARpn1x2_MYSA29_trans_leak = MYSARpn1x2_MYSA29_trans_leak * Surface_area_MYSA *1e6  ;   %nA

plot(time , MYSARpn1x2_MYSA29_trans_leak  , 'b' , 'LineWidth' , 1.5)
hold on

xlim([0 0.12])
title('Paranode Ileak (leftside) (nA)')
xlabel('time (ms)')
ylabel('Current (nA)')


%%  periaxonal current (Ip)  rightside

clc
clear all


Rpn0 = 429128.527578172*1e6*1e-4 ;  %ohm/micron   node

Right_vext0 = csvread('vext0_Rightside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');
time = Right_vext0(:,1)  ;

Ip = (Right_vext0(:,55) - Right_vext0(:,56)) / Rpn0   ; %mA.micron
Ip = Ip/0.5  ;  %mA
Ip = Ip*1e6  ;  %nA


figure
plot(time , Ip  , 'g' , 'LineWidth' , 1.5)
hold on


%%%%%%%%%%%%%%%%%%%%%%% MYSA Rpn1 x2

MYSARpn1x2_Right_vext0 = csvread('MYSARpn1x2_vext0_Rightside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');
time = MYSARpn1x2_Right_vext0(:,1)  ;

MYSARpn1x2_Ip = (MYSARpn1x2_Right_vext0(:,55) - MYSARpn1x2_Right_vext0(:,56)) / Rpn0   ; %mA.micron
MYSARpn1x2_Ip = MYSARpn1x2_Ip/0.5  ;  %mA
MYSARpn1x2_Ip = MYSARpn1x2_Ip*1e6  ;  %nA


plot(time , MYSARpn1x2_Ip  , 'b' , 'LineWidth' , 1.5)
hold on


xlim([0 1])
title('Periaxonal current (rightside) (nA)')
xlabel('time (ms)')
ylabel('Current (nA)')

%% periaxonal current (Ip)  leftside


clc
clear all


Rpn0 = 429128.527578172*1e6*1e-4 ;  %ohm/micron   node

Left_vext0 = csvread('vext0_Leftside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');
time = Left_vext0(:,1)  ;

Ip = (Left_vext0(:,55) - Left_vext0(:,56)) / Rpn0   ; %mA.micron
Ip = Ip/0.5  ;  %mA
Ip = Ip*1e6  ;  %nA


figure
plot(time , Ip  , 'g' , 'LineWidth' , 1.5)
hold on


%%%%%%%%%%%%%%%%%%%%%%% MYSA Rpn1 x2

MYSARpn1x2_Left_vext0 = csvread('MYSARpn1x2_vext0_Leftside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');
time = MYSARpn1x2_Left_vext0(:,1)  ;

MYSARpn1x2_Ip = (MYSARpn1x2_Left_vext0(:,55) - MYSARpn1x2_Left_vext0(:,56)) / Rpn0   ; %mA.micron
MYSARpn1x2_Ip = MYSARpn1x2_Ip/0.5  ;  %mA
MYSARpn1x2_Ip = MYSARpn1x2_Ip*1e6  ;  %nA


plot(time , MYSARpn1x2_Ip  , 'b' , 'LineWidth' , 1.5)
hold on


xlim([0 0.12])
title('Periaxonal current (leftside) (nA)')
xlabel('time (ms)')
ylabel('Current (nA)')


%%  Ileak of myelin  (rightside)

clc
clear all

mygm=0.001  ;
nl = 104 ;
xg0_MYSA  = (mygm/(nl*2))*1e-8 ;  %S/micron2


Right_vext0 = csvread('vext0_Rightside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');
Right_vext1 = csvread('vext1_Rightside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');


time = Right_vext0(:,1)  ;

myelin30_Il = (Right_vext0(:,54) - Right_vext1(:,54))* xg0_MYSA  ;  %mA/micron2

myelin30_Il = myelin30_Il* 3*pi*8 ;  %mA

myelin30_Il = myelin30_Il *1e6  ;  %nA


figure

plot(time , myelin30_Il  , 'g' , 'LineWidth' , 1.5)
hold on


xlim([0 0.12])
title('Ileak of myelin (rightside) (nA)')
xlabel('time (ms)')
ylabel('Current (nA)')




%%%%%%%%%%%%%%%%%%%%%%% MYSA Rpn1 x2

clc
clear all

mygm=0.001  ;
nl = 104 ;
xg0_MYSA  = (mygm/(nl*2))*1e-8 ;  %S/micron2


MYSARpn1x2_Right_vext0 = csvread('MYSARpn1x2_vext0_Rightside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');
MYSARpn1x2_Right_vext1 = csvread('MYSARpn1x2_vext1_Rightside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');


time = MYSARpn1x2_Right_vext0(:,1)  ;

MYSARpn1x2_myelin30_Il = (MYSARpn1x2_Right_vext0(:,54) - MYSARpn1x2_Right_vext1(:,54))* xg0_MYSA  ;  %mA/micron2

MYSARpn1x2_myelin30_Il = MYSARpn1x2_myelin30_Il* 3*pi*8 ;  %mA

MYSARpn1x2_myelin30_Il = MYSARpn1x2_myelin30_Il *1e6  ;  %nA



plot(time , MYSARpn1x2_myelin30_Il  , 'b' , 'LineWidth' , 1.5)
hold on




%%  Ileak of myelin  (leftside)


clc
clear all

mygm=0.001  ;
nl = 104 ;
xg0_MYSA  = (mygm/(nl*2))*1e-8 ;  %S/micron2


Left_vext0 = csvread('vext0_Leftside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');
Left_vext1 = csvread('vext1_Leftside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');


time = Left_vext0(:,1)  ;

myelin29_Il = (Left_vext0(:,54) - Left_vext1(:,54))* xg0_MYSA  ;  %mA/micron2

myelin29_Il = myelin29_Il* 3*pi*8 ;  %mA

myelin29_Il = myelin29_Il *1e6  ;  %nA


figure

plot(time , myelin29_Il  , 'g' , 'LineWidth' , 1.5)
hold on


xlim([0 0.12])
title('Ileak of myelin (Leftside) (nA)')
xlabel('time (ms)')
ylabel('Current (nA)')




%%%%%%%%%%%%%%%%%%%%%%% MYSA Rpn1 x2

clc
clear all

mygm=0.001  ;
nl = 104 ;
xg0_MYSA  = (mygm/(nl*2))*1e-8 ;  %S/micron2


MYSARpn1x2_Left_vext0 = csvread('MYSARpn1x2_vext0_Leftside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');
MYSARpn1x2_Right_vext1 = csvread('MYSARpn1x2_vext1_Leftside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');


time = MYSARpn1x2_Left_vext0(:,1)  ;

MYSARpn1x2_myelin29_Il = (MYSARpn1x2_Left_vext0(:,54) - MYSARpn1x2_Right_vext1(:,54))* xg0_MYSA  ;  %mA/micron2

MYSARpn1x2_myelin29_Il = MYSARpn1x2_myelin29_Il* 3*pi*8 ;  %mA

MYSARpn1x2_myelin29_Il = MYSARpn1x2_myelin29_Il *1e6  ;  %nA



plot(time , MYSARpn1x2_myelin29_Il  , 'b' , 'LineWidth' , 1.5)
hold on



%% Longitudinal current  (rightside)


clc
clear all


xr = 95769.75706051444*1e6*1e-4 ;   %ohm/micron  node

Right_vext1 = csvread('vext1_Rightside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');
time = Right_vext1(:,1)  ;

IL = (Right_vext1(:,55) - Right_vext1(:,56)) / xr   ; %mA.micron
IL = IL/0.5  ;  %mA
IL = IL*1e6  ;  %nA


figure
plot(time , IL  , 'g' , 'LineWidth' , 1.5)
hold on


%%%%%%%%%%%%%%%%%%%%%%% MYSA Rpn1 x2

MYSARpn1x2_Right_vext1 = csvread('MYSARpn1x2_vext1_Rightside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');
time = MYSARpn1x2_Right_vext1(:,1)  ;

MYSARpn1x2_IL = (MYSARpn1x2_Right_vext1(:,55) - MYSARpn1x2_Right_vext1(:,56)) / xr   ; %mA.micron
MYSARpn1x2_IL = MYSARpn1x2_IL/0.5  ;  %mA
MYSARpn1x2_IL = MYSARpn1x2_IL*1e6  ;  %nA


plot(time , MYSARpn1x2_IL  , 'b' , 'LineWidth' , 1.5)
hold on


xlim([0 0.12])
title('Longitudinal current (rightside) (nA)')
xlabel('time (ms)')
ylabel('Current (nA)')


%% Longitudinal current  (leftside)


clc
clear all


xr = 95769.75706051444*1e6*1e-4 ;   %ohm/micron  node

Left_vext1 = csvread('vext1_Leftside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');
time = Left_vext1(:,1)  ;

IL = (Left_vext1(:,55) - Left_vext1(:,56)) / xr   ; %mA.micron
IL = IL/0.5  ;  %mA
IL = IL*1e6  ;  %nA


figure
plot(time , IL  , 'g' , 'LineWidth' , 1.5)
hold on


%%%%%%%%%%%%%%%%%%%%%%% MYSA Rpn1 x2

MYSARpn1x2_Left_vext1 = csvread('MYSARpn1x2_vext1_Leftside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');
time = MYSARpn1x2_Left_vext1(:,1)  ;

MYSARpn1x2_IL = (MYSARpn1x2_Left_vext1(:,55) - MYSARpn1x2_Left_vext1(:,56)) / xr  ; %mA.micron
MYSARpn1x2_IL = MYSARpn1x2_IL/0.5  ;  %mA
MYSARpn1x2_IL = MYSARpn1x2_IL*1e6  ;  %nA


plot(time , MYSARpn1x2_IL  , 'b' , 'LineWidth' , 1.5)
hold on


xlim([0 0.12])
title('Longitudinal current (leftside) (nA)')
xlabel('time (ms)')
ylabel('Current (nA)')



%% I transverse 

XG =  7.08e+04 *1e-8  ;   %S/micron2

Left_vext0 = csvread('vext0_Leftside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');
Left_vext1 = csvread('vext1_Leftside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');

time = Left_vext0(:,1) ;


IT = (Left_vext1(:,56) - Left_vext0(:,56)) * XG  ; %mA/micron2

IT = IT* 2.594156*pi ;  %mA
IT = IT*1e6 ; %nA


figure
plot(time , IT  , 'g' , 'LineWidth' , 1.5)
hold on


xlim([0 0.12])
title('IT ( (nA)')
xlabel('time (ms)')
ylabel('Current (nA)')


%%%%%%%%%%%%%%%%%%%%%%% MYSA Rpn1 x2

MYSARpn1x2_Left_vext0 = csvread('MYSARpn1x2_vext0_Leftside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');
MYSARpn1x2_Left_vext1 = csvread('MYSARpn1x2_vext1_Leftside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');



MYSARpn1x2_IT = (MYSARpn1x2_Left_vext1(:,56) - MYSARpn1x2_Left_vext0(:,56)) * XG  ; %mA/micron2

MYSARpn1x2_IT = MYSARpn1x2_IT* 2.594156*pi ;  %mA
MYSARpn1x2_IT = MYSARpn1x2_IT*1e6 ; %nA



plot(time , MYSARpn1x2_IT  , 'b' , 'LineWidth' , 1.5)
hold on




%%  I boundary 

clc
clear all


Surface_area_node = 2.59415600000000*pi ;  %micron2
Right_vext1 = csvread('vext1_Rightside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');

time = Right_vext1(:,1)  ;


B =  csvread('Boundary_stimulateonlyAbeta0_edgedist0.1_.csv');


I_boundary = (B(:,2)-Right_vext1(:,56))*16.1*1e-8   ;
I_boundary = I_boundary * Surface_area_node*1e6  ; 


figure
plot(time , I_boundary ,'b' , 'LineWidth' , 2)
hold on



%%%%%%%%%%%%%%%%%%%  MYSA Rpn1 x2


Surface_area_node = 2.59415600000000*pi ;  %micron2
MYSARpn1x2_Right_vext1 = csvread('MYSARpn1x2_vext1_Rightside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');

time = MYSARpn1x2_Right_vext1(:,1)  ;


MYSARpn1x2_B =  csvread('MYSARpn1x2_Boundary_stimulateonlyAbeta0_edgedist0.1_.csv');


MYSARpn1x2_I_boundary = (MYSARpn1x2_B(:,2) - MYSARpn1x2_Right_vext1(:,56))*16.1*1e-8   ;
MYSARpn1x2_I_boundary = MYSARpn1x2_I_boundary * Surface_area_node*1e6  ; 


plot(time , MYSARpn1x2_I_boundary ,'g' , 'LineWidth' , 2)


xlabel('time(ms)')
ylabel('I-boundary (nA)')
xlim([0 0.12])


%%  node ina

clc
clear all

Surface_area_node = 2.59415600000000*pi*1e-8 ;  %cm2

node_inaa = csvread('ina_Node15Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');  %mA/cm2

time = node_inaa(:,1)  ;

node_ina = node_inaa(:,2)*Surface_area_node*1e6  ;


figure
plot(time , -node_ina ,'g' , 'LineWidth' , 2)
xlabel('time(ms)')
ylabel('node,ina (nA)')
xlim([0 0.12])
hold on


%%%%%%%%%%%%%%%%%%%% MYSA Rpn1 *2

Surface_area_node = 2.59415600000000*pi*1e-8 ;  %cm2

MYSARpn1x2_node_inaa = csvread('MYSARpn1x2_ina_Node15Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');  %mA/cm2

time = MYSARpn1x2_node_inaa(:,1)  ;

MYSARpn1x2_node_ina = MYSARpn1x2_node_inaa(:,2)*Surface_area_node*1e6  ;



plot(time , -MYSARpn1x2_node_ina , 'b' , 'LineWidth' , 2)
xlabel('time(ms)')
ylabel('node,ina (nA)')



%% current from a nearby fiber


clc
clear all

Surface_area_node = 2.59415600000000*pi ;  %micron2

XG =  7.08e+04 *1e-8  ;   %S/micron2

Right_vext1 = csvread('vext1_Rightside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');

time = Right_vext1(:,1)  ;

N =  csvread('vext1_Abeta1_stimulateonlyAbeta0_edgedist0.1_.csv');


I_nearby = (Right_vext1(:,56)-N(:,2))*XG   ;
I_nearby = I_nearby * Surface_area_node*1e6  ; 

figure
plot(time , -I_nearby ,'g' , 'LineWidth' , 2)
hold on

xlabel('time(ms)')
ylabel('I-nearby (nA)')
xlim([0 1])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MYSA Rpn1 *2

clc
clear all

Surface_area_node = 2.59415600000000*pi ;  %micron2


XG =  7.08e+04 *1e-8  ;   %S/micron2

MYSARpn1x2_Right_vext1 = csvread('MYSARpn1x2_vext1_Rightside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');

time = MYSARpn1x2_Right_vext1(:,1)  ;

MYSARpn1x2_N =  csvread('MYSARpn1x2_vext1_Abeta1_stimulateonlyAbeta0_edgedist0.1_.csv');


MYSARpn1x2_I_nearby = (MYSARpn1x2_Right_vext1(:,56)-MYSARpn1x2_N(:,2))*XG   ;
MYSARpn1x2_I_nearby = MYSARpn1x2_I_nearby * Surface_area_node*1e6  ; 


plot(time , -MYSARpn1x2_I_nearby ,'b' , 'LineWidth' , 2)
hold on

xlabel('time(ms)')
ylabel('I-nearby (nA)')







%%  Original:  Ip(right) + Ip(left) + It

clc
clear all



Rpn0 = 429128.527578172*1e6*1e-4 ;  %ohm/micron   node

Right_vext0 = csvread('vext0_Rightside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');
time = Right_vext0(:,1)  ;

Right_Ip = (Right_vext0(:,55) - Right_vext0(:,56)) / Rpn0   ; %mA.micron
Right_Ip = Right_Ip/0.5  ;  %mA
Right_Ip = Right_Ip*1e6  ;  %nA



Rpn0 = 429128.527578172*1e6*1e-4 ;  %ohm/micron   node

Left_vext0 = csvread('vext0_Leftside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');


Left_Ip = (Left_vext0(:,55) - Left_vext0(:,56)) / Rpn0   ; %mA.micron
Left_Ip = Left_Ip/0.5  ;  %mA
Left_Ip = Left_Ip*1e6  ;  %nA



XG =  7.08e+04 *1e-8  ;   %S/micron2

Left_vext0 = csvread('vext0_Leftside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');
Left_vext1 = csvread('vext1_Leftside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');




IT = (Left_vext1(:,56) - Left_vext0(:,56)) * XG  ; %mA/micron2

IT = IT* 2.594156*pi ;  %mA
IT = IT*1e6 ; %nA


I_sum = Left_Ip + Right_Ip + IT  ;


figure
plot(time , I_sum ,'g' , 'LineWidth' , 2)
hold on

xlim([0 0.12])
title('Sum current (rightside) (nA)')
xlabel('time (ms)')
ylabel('Current (nA)')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MYSA Rpn1x2    Ip(right) + Ip(left) + It



MYSARpn1x2_Right_vext0 = csvread('MYSARpn1x2_vext0_Rightside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');
time = MYSARpn1x2_Right_vext0(:,1)  ;

Right_MYSARpn1x2_Ip = (MYSARpn1x2_Right_vext0(:,55) - MYSARpn1x2_Right_vext0(:,56)) / Rpn0   ; %mA.micron
Right_MYSARpn1x2_Ip = Right_MYSARpn1x2_Ip/0.5  ;  %mA
Right_MYSARpn1x2_Ip = Right_MYSARpn1x2_Ip*1e6  ;  %nA




MYSARpn1x2_Left_vext0 = csvread('MYSARpn1x2_vext0_Leftside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');
time = MYSARpn1x2_Left_vext0(:,1)  ;

Left_MYSARpn1x2_Ip = (MYSARpn1x2_Left_vext0(:,55) - MYSARpn1x2_Left_vext0(:,56)) / Rpn0   ; %mA.micron
Left_MYSARpn1x2_Ip = Left_MYSARpn1x2_Ip/0.5  ;  %mA
Left_MYSARpn1x2_Ip = Left_MYSARpn1x2_Ip*1e6  ;  %nA


MYSARpn1x2_Left_vext0 = csvread('MYSARpn1x2_vext0_Leftside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');
MYSARpn1x2_Left_vext1 = csvread('MYSARpn1x2_vext1_Leftside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');



MYSARpn1x2_IT = (MYSARpn1x2_Left_vext1(:,56) - MYSARpn1x2_Left_vext0(:,56)) * XG  ; %mA/micron2

MYSARpn1x2_IT = MYSARpn1x2_IT* 2.594156*pi ;  %mA
MYSARpn1x2_IT = MYSARpn1x2_IT*1e6 ; %nA


MYSARpn1x2_I_sum = Left_MYSARpn1x2_Ip + Right_MYSARpn1x2_Ip + MYSARpn1x2_IT  ;

plot(time , MYSARpn1x2_I_sum ,'b' , 'LineWidth' , 2)

%% validating current calculations:  IT = IL (right) + IL (left) +IB + Inearby


clc
clear all
% close all

XG =  7.08e+04 *1e-8  ;   %S/micron2

Left_vext0 = csvread('vext0_Leftside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');
Left_vext1 = csvread('vext1_Leftside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');




IT = (Left_vext1(:,56) - Left_vext0(:,56)) * XG  ; %mA/micron2

IT = IT* 2.594156*pi ;  %mA
IT = IT*1e6 ; %nA



xr = 95769.75706051444*1e6*1e-4 ;   %ohm/micron  node

Right_vext1 = csvread('vext1_Rightside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');
time = Right_vext1(:,1)  ;

IL_right = (Right_vext1(:,55) - Right_vext1(:,56)) / xr   ; %mA.micron
IL_right = IL_right/0.5  ;  %mA
IL_right = IL_right*1e6  ;  %nA


xr = 95769.75706051444*1e6*1e-4 ;   %ohm/micron  node

Left_vext1 = csvread('vext1_Leftside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');


IL_left = (Left_vext1(:,55) - Left_vext1(:,56)) / xr   ; %mA.micron
IL_left = IL_left/0.5  ;  %mA
IL_left = IL_left*1e6  ;  %nA


Surface_area_node = 2.59415600000000*pi ;  %micron2

XG =  7.08e+04 *1e-8  ;   %S/micron2

Right_vext1 = csvread('vext1_Rightside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');


N =  csvread('vext1_Abeta1_stimulateonlyAbeta0_edgedist0.1_.csv');


I_nearby = (Right_vext1(:,56)-N(:,2))*XG   ;
I_nearby = I_nearby * Surface_area_node*1e6  ;


Surface_area_node = 2.59415600000000*pi ;  %micron2
Right_vext1 = csvread('vext1_Rightside_Abeta0_stimulateonlyAbeta0_edgedist0.1_.csv');




B =  csvread('Boundary_stimulateonlyAbeta0_edgedist0.1_.csv');


I_boundary = (B(:,2)-Right_vext1(:,56))*16.1*1e-8   ;
I_boundary = I_boundary * Surface_area_node*1e6  ; 



figure
plot( time , IT , 'm'  , 'LineWidth' , 2)
hold on

plot( time , IL_right + IL_left - I_nearby + I_boundary , 'c'  , 'LineWidth' , 2)
xlim([0 1])






