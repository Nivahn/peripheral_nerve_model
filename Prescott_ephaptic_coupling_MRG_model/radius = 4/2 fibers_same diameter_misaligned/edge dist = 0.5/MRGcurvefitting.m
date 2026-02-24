clc
clear all
close all
%%
fiberD= [5.7, 7.3, 8.7, 10.0, 11.5, 12.8, 14.0, 15.0, 16.0] ;

nodelength=1 ;
paralength1=3 ;

g=[0.605 0.630 0.661 0.690 0.700 0.719 0.739 0.767 0.791] ;
axonD= [3.4 4.6 5.8 6.9 8.1 9.2 10.4 11.5 12.7] ;
nodeD=[1.9 2.4 2.8 3.3 3.7 4.2 4.7 5.0 5.5] ;
paraD1=[1.9 2.4 2.8 3.3 3.7 4.2 4.7 5.0 5.5] ;
paraD2= [3.4 4.6 5.8 6.9 8.1 9.2 10.4 11.5 12.7] ;
deltax= [500 750 1000 1150 1250 1350 1400 1450 1500] ;
paralength2= [35 38 40 46 50 54 56 58 60] ;
nl=[80 100 110 120 130 135 140 145 150] ;

%% g  is not used 

% a = 0.01717  ;
% b = 0.5076   ;
% 
% i=1 ;
% for x=0:0.1:17
% y(1,i) = a*x+b  ;
% i=i+1 ;
% end
% 
% plot(0:0.1:17,y)
% hold on
% plot(fiberD,g,'.b')
% 
% xlabel('fiberD')
% ylabel('g')
% 
% x=9 ;
% g = a*x+b


%% axonD   (polynomial degree2)

p1 =     0.01876  ;
p2 =      0.4787  ;
p3 =      0.1204  ;


i=1 ;
for x=0:0.1:17
y(1,i) =  p1*x^2 + p2*x + p3  ;
i=i+1 ;
end

plot(0:0.1:17,y)
hold on
plot(fiberD,axonD,'.b')

xlabel('fiberD')
ylabel('axonD')


x=9 ;
axonD = p1*x^2 + p2*x + p3



%% nodeD  (polynomial degree2)

p1 =    0.006304  ;
p2 =      0.2071  ;
p3 =      0.5339  ;

i=1 ;
for x=0:0.1:17
y(1,i) =  p1*x^2 + p2*x + p3 ;
i=i+1 ;
end

plot(0:0.1:17,y)
hold on
plot(fiberD,nodeD,'.b')

xlabel('fiberD')
ylabel('nodeD')


x=9 ;
nodeD = p1*x^2 + p2*x + p3


%% deltax  (polynomial fitting gives us negative values -> so I used linear regression)


a =  92.77  ;
c =  109  ;


i=1 ;
for x=0:0.1:17
y(1,i) =  a*x+c ;
i=i+1 ;
end

plot(0:0.1:17,y)
hold on
plot(fiberD,deltax,'.b')

xlabel('fiberD')
ylabel('deltax')


x=9 ;
deltax =  a*x+c

%%  deltax  (polynomial)

 p1 =      -8.215  ;
 p2 =       272.4  ;
 p3 =      -780.2  ;



i=1 ;
for x=0:0.1:17
y(1,i) = p1*x^2 + p2*x + p3  ;
i=i+1 ;
end

plot(0:0.1:17,y)
hold on
plot(fiberD,deltax,'.b')

xlabel('fiberD')
ylabel('deltax')





%% paralength2   (polynomial degree2)

p1 =     -0.0199  ;
p2 =       3.016  ;
p3 =       17.44  ;



i=1 ;
for x=0:0.1:17
y(1,i) = p1*x^2 + p2*x + p3  ;
i=i+1 ;
end

plot(0:0.1:17,y)
hold on
plot(fiberD,paralength2,'.b')

xlabel('fiberD')
ylabel('paralength2')


x=9 ;
paralength2 = p1*x^2 + p2*x + p3





%% nl    (polynomial degree2)

       p1 =      -0.389 ;
       p2 =       14.88  ;
       p3 =       9.721  ;
i=1 ;
for x=0:0.1:17
y(1,i) =  p1*x^2 + p2*x + p3  ;
i=i+1 ;
end

plot(0:0.1:17,y)
hold on
plot(fiberD,nl,'.b')

xlabel('fiberD')
ylabel('nl')

x=9 ;
nl =  p1*x^2 + p2*x + p3




