
clc
clear all
close all


%% boundary = 0.017     dist=0.029    
% Abeta#4
x4=[0 1 2 3 4 5 6 7 8 9 10] ;
y4=[15.2295  18.5634  18.1718  21.8457  18.5096  24.3349  19.0735  22.2737  18.6681  18.8407  8.93171] ;


% Abeta#5
x5=[0 1 2 3 4 5 6 7 8 9 10] ;
y5= [11.6788  3.50992  1.94536  2.83677  1.85506  1.66078  3.33581  3.32711  5.06387  7.69957 5.67181 ] ;


% Abeta#7
x7=[0 1 2 3 4 5 6 7 8 9 10] ;
y7=[11.6032 3.460383 1.90725 2.741489 1.842582  1.763512396  3.403470122  3.399821991  5.235668488  7.709054019  5.581916656]  ;

 
% Abeta#9
x9=[0 1 2 3 4 5 6 7 ] ;
y9=[13.659 3.55006  7.664  3.16563  3.96797  6.5762  2.2478  5.10934] ;


%Abeta#12
x12=[0 1 2 3 4 5 6 7 8 ] ;
y12=[13.2862927  3.570024705  3.041370636  9.82711207  5.230769265  2.951681434  6.325436575  4.957065225  3.048962951 ] ;

 
%Abeta#15
x15=[0 1 2 3 4 5 6 7 8 9 10] ;
y15=[11.94934924  7.092400565  6.644005431  6.524317826  5.060213144  4.474416922  4.350261133  3.297594225  1.877277595  1.639925577   1.699706093] ;                  

%Abeta#17
x17=[0 1 2 3 4 5 6 7 8 9 10] ;
y17=[11.742734  6.311058059  4.663662582  4.048240217   1.955292653  3.516711456  2.240724002  2.093808423  1.590957128   1.275412868   2.816560345] ;


%Abeta#18
x18=[0 1 2 3 4 5 6 7 8 9 ] ;
y18=[12.21307294  3.480898719  2.30353868  3.386461802  4.817269805  7.952117281  8.827661772  4.865453019  2.356172593  2.829339201 ] ;




figure
plot(x4,y4 , '-o', 'LineWidth', 1)
hold on
plot(x5,y5 , '-o', 'LineWidth', 1)
hold on
plot(x7,y7 , '-o', 'LineWidth', 1)
hold on
plot(x9,y9 , '-o', 'LineWidth', 1)
hold on
plot(x12,y12 , '-o', 'LineWidth', 1)
hold on
plot(x15,y15 , '-o', 'LineWidth', 1)
hold on
plot(x17,y17 , '-o', 'LineWidth', 1)
hold on
plot(x18,y18 , 'o-k' , 'LineWidth', 1)


legend('Abeta#4' , 'Abeta#5' , 'Abeta#7' , 'Abeta#9' , 'Abeta#12' , 'Abeta#15' , 'Abeta#17' , 'Abeta#18' )
xlabel('node')
ylabel('peak-to-peak deflection (mV)')






%% boundary = 0.017     dist=0.029      Abeta#5     node#5
clc
clear all
close all

x=[0.029  0.25  1  2  5  10 ] ;

y1=[0.0022 0.00242 0.0025 0.0025 0.00247 0.00247] ;
 y1=log10(y1) ;
y2=[0.01366 0.01725 0.01812 0.01756 0.01549 0.01496] ;
y2=log10(y2) ;
y3=[0.08912 0.10796 0.10885 0.12228 0.27069 0.4753] ;
y3=log10(y3) ;
y4=[0.434412  0.65986  1.23048  1.85481  2.68826  2.52104]  ;
 y4=log10(y4) ;
y5=[1.660781  2.17318  3.34833  4.26616  4.85101  3.85767] ;
 y5=log10(y5) ;
y6=[109.462   112.781  3.8988   4.82508  5.28507  4.08536]  ;
% y6=log10(y6) ;


figure
plot(x,y1,'*-r', 'LineWidth', 2)
hold on
plot(x,y2,'*-b', 'LineWidth', 2)
 hold on
plot(x,y3,'*-g', 'LineWidth', 2)
hold on
plot(x,y4,'*-m', 'LineWidth', 2)
hold on
plot(x,y5,'*-k' , 'LineWidth', 2)
% hold on
% plot(x,y6,'*-c' , 'LineWidth', 2)

xlim([0 10])
%ylim([0.002 0.0028])
legend('g-boundary (S/cm2) = 171.1'  , 'g-boundary (S/cm2) = 17.1' , 'g-boundary (S/cm2) = 1.71', 'g-boundary (S/cm2) = 0.171' , 'g-boundary (S/cm2) = 0.0171' , 'g-boundary (S/cm2) = 0.00171')

xlabel('Average edge-to-edge ditance between Abeta#2 and its nearest-nearby fibers ')
ylabel('peak-to-peak voltage deflection (mV)')

title('Abeta#5, node#6')

%% boundary = 0.017     dist=0.029      Abeta#5     node#0
clc
clear all
close all

x=[0.029  0.25  1  2  5  10 ] ;

y1=[0.01802  0.01845  0.01875  0.01877  0.01876  0.01872] ;
y2=[0.17155  0.1762  0.17798  0.17754  0.17539  0.17092] ;
y3=[1.53894  1.57519  1.5542  1.51621  1.36865  1.11854] ;
y4=[7.15515  7.27172  6.78966  6.08682  4.44941  2.80129]  ;
y5=[11.6788  11.8031  10.5681  9.05602  6.01297  3.51555] ;
y6=[118.542  117.637  11.3075  9.64165  6.31165  3.76117]  ;


figure
plot(x,y1,'*-r', 'LineWidth', 2)
hold on
plot(x,y2,'*-b', 'LineWidth', 2)
 hold on
plot(x,y3,'*-g', 'LineWidth', 2)
hold on
plot(x,y4,'*-m', 'LineWidth', 2)
hold on
plot(x,y5,'*-k' , 'LineWidth', 2)
hold on
plot(x,y6,'*-c' , 'LineWidth', 2)



legend('g-boundary (S/cm2) = 171.1'  , 'g-boundary (S/cm2) = 17.1' , 'g-boundary (S/cm2) = 1.71', 'g-boundary (S/cm2) = 0.171' , 'g-boundary (S/cm2) = 0.0171' , 'g-boundary (S/cm2) = 0.00171')

xlabel('Average edge-to-edge ditance between Abeta#2 and its nearest-nearby fibers ')
ylabel('peak-to-peak voltage deflection (mV)')

title('Abeta#5, node#0')




%
%% boundary = 0.017     dist=0.029      Abeta#4     node#6
clc
clear all
close all

x=[0.029  0.25  1  2  5  10 ] ;

y1=[0.11485  0.11555  0.11618  0.1152  0.11353  0.10907] ;
y2=[1.09792  1.10015  1.07762  1.04605  0.92818  0.71932] ;
y3=[7.42977  7.34311  6.61331  5.6714  3.70704  2.17658] ;
y4=[19.3034  18.3232  14.8108  11.7279  7.02737  3.93464]  ;
y5=[24.3349  22.896  18.1467  14.2671  8.56235  4.77691] ;
y6=[115.186  111.753  18.6569  14.7386  8.86729  4.92468]  ;


figure
plot(x,y1,'*-r', 'LineWidth', 2)
hold on
plot(x,y2,'*-b', 'LineWidth', 2)
 hold on
plot(x,y3,'*-g', 'LineWidth', 2)
hold on
plot(x,y4,'*-m', 'LineWidth', 2)
hold on
plot(x,y5,'*-k' , 'LineWidth', 2)
hold on
plot(x,y6,'*-c' , 'LineWidth', 2)



legend('g-boundary (S/cm2) = 171.1'  , 'g-boundary (S/cm2) = 17.1' , 'g-boundary (S/cm2) = 1.71', 'g-boundary (S/cm2) = 0.171' , 'g-boundary (S/cm2) = 0.0171' , 'g-boundary (S/cm2) = 0.00171')

xlabel('Average edge-to-edge ditance between Abeta#2 and its nearest-nearby fibers (micron)')
ylabel('peak-to-peak voltage deflection (mV)')

title('Abeta#4, node#6')



%% boundary = 0.017     dist=0.029      Abeta#4     node#0
clc
clear all
close all

x=[0.029  0.25  1  2  5  10 ] ;

y1=[0.04504  0.04479  0.04357  0.04357  0.04243  0.03914] ;
y2=[0.42987  0.42557  0.39721  0.37773  0.31304  0.23882] ;
y3=[3.04892  2.86037  2.42926  2.09283  1.56089  1.17099] ;
y4=[10.4173  9.39466  7.70647  6.54091  4.45682  2.69451]  ;
y5=[15.2295  13.722  11.1092  9.15664  5.84945  3.27294] ;
y6=[115.406  115.441  11.7358  9.60911  6.03115  3.35747]  ;


figure
plot(x,y1,'*-r', 'LineWidth', 2)
hold on
plot(x,y2,'*-b', 'LineWidth', 2)
 hold on
plot(x,y3,'*-g', 'LineWidth', 2)
hold on
plot(x,y4,'*-m', 'LineWidth', 2)
hold on
plot(x,y5,'*-k' , 'LineWidth', 2)
hold on
plot(x,y6,'*-c' , 'LineWidth', 2)



legend('g-boundary (S/cm2) = 171.1'  , 'g-boundary (S/cm2) = 17.1' , 'g-boundary (S/cm2) = 1.71', 'g-boundary (S/cm2) = 0.171' , 'g-boundary (S/cm2) = 0.0171' , 'g-boundary (S/cm2) = 0.00171')

xlabel('Average edge-to-edge ditance between Abeta#2 and its nearest-nearby fibers (micron)')
ylabel('peak-to-peak voltage deflection (mV)')

title('Abeta#4, node#0')


%%



clc
clear all
close all


x = [ 3.83e5/2    3.83e5/2    3.83e5/2    3.83e5/2] ;
x=log10(x) ;
y = [335369.13/2   335369.13       335369.13*2      335369.13*2] ;
y=log10(y) ;
c = [ 0.01587  0.01366  0  0.01109 ] ;
scatter(x,y,150,c,'filled')
colorbar
% colormap jet
hold on


x = [3.83e5   3.83e5    3.83e5    3.83e5] ;
x=log10(x) ;
y = [335369.13/2   335369.13    335369.13*2    335369.13*2] ;
y=log10(y) ;
c = [0.01587  0.01366   0    0.01109];
scatter(x,y,150,c,'filled')
colorbar
colormap parula
hold on



x = [3.83e5*2   3.83e5*2   3.83e5*2   3.83e5*2] ;
x=log10(x) ;
y = [335369.13/2   335369.13     335369.13*2   335369.13*2] ;
y=log10(y) ;
c = [0.01587  0.01366    0     0.01109];
scatter(x,y,150,c,'filled')
colorbar
colormap parula
hold on





caxis([0 0.02])
title('g-boundary=17.1   ,  dist=0.03')
xlabel('Transvers-g between Abeta2 and Abeta5 (S/cm2)')
ylabel('xraxial[1], Abeta2 (Mohm/cm)')


%%

clc
clear all
close all


x = [ 4.05e3/2    4.05e3/2    4.05e3/2    4.05e3/2] ;
x=log10(x) ;
y = [9284.41/2   9284.41    9284.41*2      9284.41*2] ;
y=log10(y) ;
c = [ 0.0178 0.01813  0   0.018 ] ;
scatter(x,y,150,c,'filled')
colorbar
% colormap jet
hold on


x = [ 4.05e3    4.05e3    4.05e3    4.05e3] ;
x=log10(x) ;
y = [9284.41/2   9284.41    9284.41*2      9284.41*2] ;
y=log10(y) ;
c = [0.01767  0.01812  0   0.018];
scatter(x,y,150,c,'filled')
colorbar
colormap parula
hold on



x = [ 4.05e3*2    4.05e3*2    4.05e3*2    4.05e3*2] ;
x=log10(x) ;
y = [9284.41/2   9284.41    9284.41*2      9284.41*2] ;
y=log10(y) ;
c = [0.01767 0.01812  0  0.01799];
scatter(x,y,150,c,'filled')
colorbar
colormap parula
hold on





caxis([0 0.02])
title('g-boundary=17.1   ,  dist=1')
xlabel('Transvers-g between Abeta2 and Abeta5 (S/cm2)')
ylabel('xraxial[1], Abeta2 (Mohm/cm)')



%%

clc
clear all
close all


x = [ 798/2    798/2    798/2    798/2] ;
x=log10(x) ;
y = [1496/2   1496    1496*2      1496*2] ;
y=log10(y) ;
c = [ 0.01354  0.01554  0  0.0169 ] ;
scatter(x,y,150,c,'filled')
colorbar
% colormap jet
hold on


x = [ 798    798    798   798] ;
x=log10(x) ;
y = [1496/2   1496    1496*2      1496*2] ;
y=log10(y) ;
c = [0.01347  0.01549   0    0.01687];
scatter(x,y,150,c,'filled')
colorbar
colormap parula
hold on



x = [ 798*2    798*2    798*2    798*2] ;
x=log10(x) ;
y = [1496/2   1496    1496*2      1496*2] ;
y=log10(y) ;
c = [0.01343  0.01545    0  0.01684];
scatter(x,y,150,c,'filled')
colorbar
colormap parula
hold on





caxis([0 0.02])
title('g-boundary=17.1   ,  dist=5')
xlabel('Transvers-g between Abeta2 and Abeta5 (S/cm2)')
ylabel('xraxial[1], Abeta2 (Mohm/cm)')


%%    boundar=17.1

clc
clear all
close all



xx=[0.029  0.25  1  2  5  10 ] ;

y4=[1.097  1.1001  1.077  1.046  0.928  0.719]  ;
%y4=log10(y4) ;
y5=[0.013  0.0172  0.018  0.0175  0.0154  0.0149]  ;
%y5=log10(y5) ;
y7=[0.0171  0.0183  0.0181  0.01744  0.0154  0.0175] ;
%y7=log10(y7) ;
y9=[0.0118  0.0147  0.0152  0.01552  0.0226  0.0429] ;
%y9=log10(y9) ;
y12=[0.00574  0.0193  0.0230  0.02299  0.02998  0.0676]  ;
%y12=log10(y12) ;
y15=[0.0089 0.0200 0.0204  0.0246  0.056  0.0959] ;
%y15=log10(y15) ;
y17=[0.00887  0.0246  0.0303  0.0297  0.0244  0.018] ;
%y17=log10(y17) ;
y18=[0.008  0.0178  0.029  0.0439  0.09232  0.14813]  ;
%y18=log10(y18) ;

figure
plot(xx,y4,'*-r', 'LineWidth', 2)
hold on
plot(xx,y5,'*-b', 'LineWidth', 2)
hold on
plot(xx,y7,'*-g', 'LineWidth', 2)
hold on
plot(xx,y9,'*-m', 'LineWidth', 2)
hold on
plot(xx,y12,'*-k' , 'LineWidth', 2)
hold on
plot(xx,y15,'*-c' , 'LineWidth', 2)
hold on
plot(xx,y17,'*-y' , 'LineWidth', 2)
hold on
plot(xx,y18,	'-*','color', [0.4940 0.1840 0.5560],  'LineWidth', 2)


%ylim([-3 3])
%legend('4' , '5' , '7' , '9' , '12' , '15' , '17' ,'18' )
title('boundary = 17.1')
xlabel('Average edge-to-edge distance between Abeta#2 and its nearest-nearby fibers')
ylabel('peak-to-peak voltage deflection (mV)')



%% boundary = 171

clc
clear all
close all


xx=[0.029  0.25  1  2  5  10 ] ;

y4=[0.1148  0.1155  0.1161  0.1152  0.1135  0.109]  ;
y5=[0.0022  0.00242  0.0025  0.0025  0.00247  0.00242]  ;
y7=[0.00246  0.00258  0.00256  0.00257  0.00256  0.00251] ;
y9=[0.00205  0.00225  0.00224  0.00225  0.00228  0.00235] ;
y12=[0.00245  0.00299  0.00341  0.00349  0.00351  0.00341]  ;
y15=[0.00239  0.00343  0.00343  0.00342  0.00321  0.00313] ;
y17=[0.00252  0.00323  0.00379  0.00389  0.00392  0.00386] ;
y18=[0.00243  0.00215  0.00223  0.00229 0.00254  0.00413]  ;


y4=log10(y4) ;
y5=log10(y5) ;
y7=log10(y7) ;
y9=log10(y9) ;
y12=log10(y12) ;
y15=log10(y15) ;
y17=log10(y17) ;
y18=log10(y18) ;





figure
plot(xx,y4,'*-r', 'LineWidth', 2)
hold on
plot(xx,y5,'*-b', 'LineWidth', 2)
 hold on
plot(xx,y7,'*-g', 'LineWidth', 2)
hold on
plot(xx,y9,'*-m', 'LineWidth', 2)
hold on
plot(xx,y12,'*-k' , 'LineWidth', 2)
hold on
plot(xx,y15,'*-c' , 'LineWidth', 2)
hold on
plot(xx,y17,'*-y' , 'LineWidth', 2)
hold on
plot(xx,y18,	'-*','color', [0.4940 0.1840 0.5560],  'LineWidth', 2)


%ylim([-3 3])
%legend('4' , '5' , '7' , '9' , '12' , '15' , '17' ,'18' )
title('boundary = 171')
xlabel('Average edge-to-edge distance between Abeta#2 and its nearest-nearby fibers')
ylabel('peak-to-peak voltage deflection (mV)')

 %% boundary = 1.71

clc
clear all
close all



xx=[0.029  0.25  1  2  5  10 ] ;

y4=[7.429  7.343  6.613  5.671  3.707  2.1765]  ;
y5=[0.089  0.107  0.1088  0.1222  0.2706  0.475]  ;
y7=[0.11763  0.11718  0.11471  0.1417  0.2809  0.479] ;
y9=[0.1012  0.156  0.2539  0.3717  0.6378  0.8113] ;
y12=[0.237  0.206  0.377  0.572  0.872  0.907]  ;
y15=[0.249  0.317  0.552  0.7519  0.968  0.9156] ;
y17=[0.0717  0.135  0.152  0.15013  0.402  0.561] ;
y18=[0.486  0.648  0.977  1.274  1.543  1.350]  ;



y4=log10(y4) ;
y5=log10(y5) ;
y7=log10(y7) ;
y9=log10(y9) ;
y12=log10(y12) ;
y15=log10(y15) ;
y17=log10(y17) ;
y18=log10(y18) ;





figure
plot(xx,y4,'*-r', 'LineWidth', 2)
hold on
plot(xx,y5,'*-b', 'LineWidth', 2)
 hold on
plot(xx,y7,'*-g', 'LineWidth', 2)
hold on
plot(xx,y9,'*-m', 'LineWidth', 2)
hold on
plot(xx,y12,'*-k' , 'LineWidth', 2)
hold on
plot(xx,y15,'*-c' , 'LineWidth', 2)
hold on
plot(xx,y17,'*-y' , 'LineWidth', 2)
hold on
plot(xx,y18,	'-*','color', [0.4940 0.1840 0.5560],  'LineWidth', 2)

%ylim([-3 3])
%legend('4' , '5' , '7' , '9' , '12' , '15' , '17' ,'18' )
title('boundary = 1.71')
xlabel('Average edge-to-edge distance between Abeta#2 and its nearest-nearby fibers')
ylabel('peak-to-peak voltage deflection (mV)')


%% boundary = 0.171

clc
clear all
close all



xx=[0.029  0.25  1  2  5  10 ] ;

y4=[19.3034  18.323  14.810  11.727  7.027  3.934]  ;
y5=[0.434  0.659  1.230  1.854  2.688  2.521]  ;
y7=[0.559  0.741  1.266  1.876  2.692  2.522] ;
y9=[1.538  1.9522  2.772  3.437  3.959  3.384] ;
y12=[2.530  2.617  3.402  3.830  3.786  3.020]  ;
y15=[2.475  2.879  3.607  3.875  3.518  2.578] ;
y17=[1.419  1.268  1.703  2.163  2.609  2.3618] ;
y18=[4.449  5.089  5.995  6.230  5.253  3.5508]  ;



y4=log10(y4) ;
y5=log10(y5) ;
y7=log10(y7) ;
y9=log10(y9) ;
y12=log10(y12) ;
y15=log10(y15) ;
y17=log10(y17) ;
y18=log10(y18) ;






figure
plot(xx,y4,'*-r', 'LineWidth', 2)
hold on
plot(xx,y5,'*-b', 'LineWidth', 2)
 hold on
plot(xx,y7,'*-g', 'LineWidth', 2)
hold on
plot(xx,y9,'*-m', 'LineWidth', 2)
hold on
plot(xx,y12,'*-k' , 'LineWidth', 2)
hold on
plot(xx,y15,'*-c' , 'LineWidth', 2)
hold on
plot(xx,y17,'*-y' , 'LineWidth', 2)
hold on
plot(xx,y18,	'-*','color', [0.4940 0.1840 0.5560],  'LineWidth', 2)

%ylim([-3 3])
%legend('4' , '5' , '7' , '9' , '12' , '15' , '17' ,'18' )
title('boundary = 0.171')
xlabel('Average edge-to-edge distance between Abeta#2 and its nearest-nearby fibers')
ylabel('peak-to-peak voltage deflection (mV)')

%% boundary = 0.0171

clc
clear all
close all



xx=[0.029  0.25  1  2  5  10 ] ;

y4=[24.3349  22.896  18.1467  14.2671  8.562  4.77691]  ;
y5=[1.660  2.173  3.348  4.266  4.851  3.857]  ;
y7=[1.763  2.2282  3.372  4.276  4.854  3.858] ;
y9=[3.967  4.6200  5.872  6.662  6.687  5.0566] ;
y12=[5.230  5.523  6.429  6.75403  6.219  4.58183]  ;
y15=[4.474  5.130  5.900  5.942  5.070  3.621] ;
y17=[3.516  3.3199  3.919  4.444  4.658  3.628] ;
y18=[7.952  8.777  9.567  9.366  7.417  4.787]  ;



y4=log10(y4) ;
y5=log10(y5) ;
y7=log10(y7) ;
y9=log10(y9) ;
y12=log10(y12) ;
y15=log10(y15) ;
y17=log10(y17) ;
y18=log10(y18) ;




figure
plot(xx,y4,'*-r', 'LineWidth', 2)
hold on
plot(xx,y5,'*-b', 'LineWidth', 2)
 hold on
plot(xx,y7,'*-g', 'LineWidth', 2)
hold on
plot(xx,y9,'*-m', 'LineWidth', 2)
hold on
plot(xx,y12,'*-k' , 'LineWidth', 2)
hold on
plot(xx,y15,'*-c' , 'LineWidth', 2)
hold on
plot(xx,y17,'*-y' , 'LineWidth', 2)
hold on
plot(xx,y18,	'-*','color', [0.4940 0.1840 0.5560],  'LineWidth', 2)

%ylim([-3 3])
%legend('4' , '5' , '7' , '9' , '12' , '15' , '17' ,'18' )
title('boundary = 0.0171')
xlabel('Average edge-to-edge distance between Abeta#2 and its nearest-nearby fibers')
ylabel('peak-to-peak voltage deflection (mV)')

%% boundary = 0.00171

clc
clear all
close all



xx=[0.029  0.25  1  2  5  10 ] ;

y4=[115.186  111.753  18.656  14.738  8.867  4.924]  ;
y5=[109.462  112.781  3.8988  4.825  5.285  4.085]  ;
y7=[109.181  112.676  3.9168  4.8333  5.288  4.086] ;
y9=[110.136  109.661  6.622  7.39607  7.232  5.3387] ;
y12=[113.72  110.466  7.202  7.4116  6.716  4.866]  ;
y15=[108.408  107.504  6.339  6.362  5.342  3.829] ;
y17=[111.024  106.704  4.449  4.958  5.080  3.859] ;
y18=[105.651  104.948  10.273  9.959  7.825  4.992]  ;




y4=log10(y4) ;
y5=log10(y5) ;
y7=log10(y7) ;
y9=log10(y9) ;
y12=log10(y12) ;
y15=log10(y15) ;
y17=log10(y17) ;
y18=log10(y18) ;




figure
plot(xx,y4,'*-r', 'LineWidth', 2)
hold on
plot(xx,y5,'*-b', 'LineWidth', 2)
 hold on
plot(xx,y7,'*-g', 'LineWidth', 2)
hold on
plot(xx,y9,'*-m', 'LineWidth', 2)
hold on
plot(xx,y12,'*-k' , 'LineWidth', 2)
hold on
plot(xx,y15,'*-c' , 'LineWidth', 2)
hold on
plot(xx,y17,'*-y' , 'LineWidth', 2)
hold on
plot(xx,y18,	'-*','color', [0.4940 0.1840 0.5560],  'LineWidth', 2)

%legend('4' , '5' , '7' , '9' , '12' , '15' , '17' ,'18' )
%ylim([-3 3])
title('boundary = 0.00171')

xlabel('Average edge-to-edge distance between Abeta#2 and its nearest-nearby fibers')
ylabel('peak-to-peak voltage deflection (mV)')



%%%%%%
%% boundary = 17.1     # Abeta4 twice more nodes 180 degree misalingment

clc
clear all
close all

xx=[0.029  0.25  1  2  5  10 ] ;

y4_360=[1.097  1.1001  1.077  1.046  0.928  0.719]  ;
y4_360=log10(y4_360) ;


y4_180=[2.2026  2.2136  2.1259  2.0123  1.629  1.0968]  ;
y4_180=log10(y4_180) ;


figure
plot(xx,y4_360,'*-r', 'LineWidth', 2)
hold on

plot(xx,y4_180,'*--r', 'LineWidth', 2)
hold on


legend('Abeta#4,  360 degree alignment'  ,  'Abeta#4, 180 degree alignment')
ylim([-3 3])
title('boundary = 17.1')

xlabel('Average edge-to-edge distance between Abeta#2 and its nearest-nearby fibers')
ylabel('peak-to-peak voltage deflection (mV)')



%%    April 1  2020

% boundary =171

clc
clear all



x=[ 3.9  4.1  4.2  4.4  4.4  4.7  5.3  5.8 ] ;   % radii of different Abeta fibers

y029=[0.1148  0.00239  0.00252  0.0022  0.00246   0.00243  0.00245  0.00205 ] ;  % distance = 0.029
y25=[0.1155   0.00343  0.00323  0.00242 0.00258   0.00215   0.00299  0.00225 ]  ;
y1= [0.1161   0.00343  0.00379  0.0025  0.00256   0.00223   0.00341  0.00224 ] ;
y2=[0.1152    0.00342  0.00389  0.0025  0.00257   0.00229   0.00349  0.00225] ;
y5=[0.1135    0.00321  0.00392  0.00247 0.00256   0.00254   0.00351  0.00228]  ;
y10=[0.109    0.00313  0.00386  0.00242 0.00251   0.00413    0.00341 0.00235 ] ;
 


y029=log10(y029) ;
y25=log10(y25) ;
y1=log10(y1) ;
y2=log10(y2) ;
y5=log10(y5) ;
y10=log10(y10) ;

figure
plot(x,y029,'*-r', 'LineWidth', 2)
hold on
plot(x,y25,'*-b', 'LineWidth', 2)
 hold on
plot(x,y1,'*-g', 'LineWidth', 2)
hold on
plot(x,y2,'*-m', 'LineWidth', 2)
hold on
plot(x,y5,'*-k' , 'LineWidth', 2)
hold on
plot(x,y10,'*-c' , 'LineWidth', 2)




%%


% boundary =0.171

clc
clear all
close all


x=[ 3.9  4.1  4.2  4.4  4.4  4.7  5.3  5.8 ] ;   % radii of different Abeta fibers

y029=[19.3034  2.475  1.419  0.434  0.559  4.449  2.530   1.537] ;  % distance = 0.029
y25=[18.323  2.879  1.268  0.659  0.741  5.089  2.617   1.952 ]  ;
y1= [14.810  3.607  1.703  1.230   1.266  5.995  3.402  2.772] ;
y2=[11.727  3.875  2.163   1.854   1.876    6.230  3.830  3.437] ;
y5=[7.027   3.518   2.609  2.688   2.692   5.253   3.786  3.959]  ;
y10=[ 3.934  2.578  2.3618  2.521   2.522  3.5508   3.020  3.384] ;
 

figure
plot(x,y029,'*-r', 'LineWidth', 2)
hold on
plot(x,y25,'*-b', 'LineWidth', 2)
 hold on
plot(x,y1,'*-g', 'LineWidth', 2)
hold on
plot(x,y2,'*-m', 'LineWidth', 2)
hold on
plot(x,y5,'*-k' , 'LineWidth', 2)
hold on
plot(x,y10,'*-c' , 'LineWidth', 2)


y4=[19.3034  18.323  14.810  11.727  7.027  3.934]  ;
y15=[2.475  2.879  3.607  3.875  3.518  2.578] ;
y17=[1.419  1.268  1.703  2.163  2.609  2.3618] ;
y5=[0.434  0.659  1.230  1.854  2.688  2.521]  ;
y7=[0.559  0.741  1.266  1.876  2.692  2.522] ;
y18=[4.449  5.089  5.995  6.230  5.253  3.5508]  ;
y12=[2.530  2.617  3.402  3.830  3.786  3.020]  ;

