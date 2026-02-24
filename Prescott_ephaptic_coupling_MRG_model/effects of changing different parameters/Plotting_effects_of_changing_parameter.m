clc
clear all
close all

%%  3 cm (36 nodes).  mean across node 10 to 20 .  duration of VClamp = 0.02


nodeD = [           25.4714       36.2389     39.7008   36.6909  ] ;
Ra = [  75.9137      52.8916       36.2389     24.0240   15.1295  ] ;
nl = [  23.7391      30.8587       36.2389     39.7132    41.7631 ] ;
fiberD = [8.5979     19.1918       36.2389     58.8677    83.7410 ] ;
%interlength = [ 14.6497   23.9940    36.2389    48.8015  54.1881  49.6869  41.8053  ] ;
interlength = [ 14.6497   23.9940    36.2389    48.8015  54.1881    ] ;
xraxil0_STINLUT = [ 35.3105        35.6374     36.2389   37.0681   37.9589      ] ;
xraxil0_MYSA = [    22.3968        27.9740     36.2389   46.4852   56.6942      ] ;


t_nodeD = [ -2  0  2 4] ;
t=[-4 -2  0  2 4] ;
t_interlength = [-4 -2  0  2 4 6 8] ;

figure
plot(t_nodeD , nodeD , '-ok' , 'LineWidth' , 1.5 )
hold on
plot(t , Ra , '-ob' , 'LineWidth' , 1.5 )
hold on
plot(t , nl , '-or' , 'LineWidth' , 1.5 )
hold on
plot(t , fiberD , '-og' , 'LineWidth' , 1.5 )
hold on
plot(t , interlength  , '-om' , 'LineWidth' , 1.5 )
hold on
plot(t , xraxil0_STINLUT  , '-oc' , 'LineWidth' , 1.5 )
hold on
plot(t , xraxil0_MYSA  , '-oy' , 'LineWidth' , 1.5 )



legend('Node Diam' , 'Axial resistivity' , 'Number of myelin sheaths' , 'Axon Diam' , 'Internode length' , 'Xraxial[0] Juxtaparanode and Internode' , 'Xraxial[0]Paranode')

ylabel('Conduction Velocity (m/s)')
xlim([-6 6])

%%



