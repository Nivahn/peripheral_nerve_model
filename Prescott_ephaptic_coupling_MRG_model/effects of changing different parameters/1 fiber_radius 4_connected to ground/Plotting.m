clc
clear all
close all


%%  1 axon, connected to ground


clc
clear all
close all

nodeD = [                  ] ;
nodeLength = [                          ]   ;
Ra = [                  ] ;
nl = [                     ] ;
fiberD = [                      ] ;
STINlength = [                   ] ;
xraxial0_FLUTSTIN = [                  ] ; 
xraixal0_MYSA = [                          ]  ;


t=[-32 -16 -8 -4 -2  0  2 4 8 16 32] ;
figure
plot(t , nodeD , '-ok' , 'LineWidth' , 1.5 )
hold on
plot(t , Ra , '-ob' , 'LineWidth' , 1.5 )
hold on
plot(t , nl , '-or' , 'LineWidth' , 1.5 )
hold on
plot(t , fiberD , '-og' , 'LineWidth' , 1.5 )
hold on
plot(t , STINlength  , '-om' , 'LineWidth' , 1.5 )


legend('Node Diam' , 'Axial resistivity' , 'Number of myelin sheaths' , 'Axon Diam' , 'Internode length')

ylabel('Conduction Velocity (m/s)')
xlim([-3 3])
ylim([ 15  65])
title('Average CV')


