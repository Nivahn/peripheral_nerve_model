clc
clear all
close all

%%

xx=1 ;



%%

Data = csvread('misaligned_v_Abeta0_stimulateBOTH_edgedist3_.csv');

time = Data(:,1) ;

[satr, soton] = size(Data) ;


figure

for i=2:soton

plot(time , Data(:,i), '-b', 'LineWidth' , 1 )
hold on
end

%xlim([0 5])
ylim([-90 40])


Data1 = Data(:,2:soton) ;   %exclude time from the data

%%


soton=soton-1 ;
number_nodes =  soton ;

%%

n = 1 ;

s=n+1:number_nodes-n ; 
S1 = zeros(1,length(s)) ;
k=1 ;



for j=n+1:number_nodes-n

clear V
clear A AA1 AA2 pks locs L1 L2 B1 B2

V = Data1(:,j) ;

[pks , locs] = findpeaks(V,'MinPeakDistance',400, 'MinPeakHeight',0)     ;



for i=1:length(pks)
     
        
  A(1,:) = V(locs(i,1)-30:locs(i,1) , 1) ;
  
  AA1 = A(A>=-20) ;
  [L1]=knnsearch(AA1',-20) ;
   B1 = AA1(1,L1) ;
       
  AA2 = A(A<=-20) ;
  [L2]=knnsearch(AA2',-20) ;
   B2 = AA2(1,L2) ;
   
  [row1]=find(V==B1) ;
  [row2]=find(V==B2) ;
  
  if length(row1)>1 
      for l=1:length(row1)
          if row1(l) > locs(i,1)-30 && row1(l) < locs(i,1)
              row11 = row1(l) ;
          end
      end
         
      
  else
      row11 = row1 ;
  end
  
  if length(row2)>1 
      for l=1:length(row2)
          if row2(l) > locs(i,1)-30 && row2(l) < locs(i,1)
              row22 = row2(l) ;
          end
      end
         
      
  else
      row22 = row2 ;
  end
  
  
  
  
  T1 = time(row11,1) ;
  T2 = time(row22,1) ;
  
  
coefficients = polyfit([T1, T2], [B1, B2], 1);
a = coefficients (1);
b = coefficients (2);

f =  -20  ;

x = (f-b)/a  ;
     

S1(1,k) = x  ;

plot(x,-20 , '*r')

  end      

k=k+1  ;

end

%%

figure


for i=1:length(S1)

plot(S1(1,i) , 1 , '.', 'Color','none')
text(S1(1,i), 1, '|', 'VerticalAlignment','middle', 'HorizontalAlignment','center', 'Color','m')
hold on


end


ylim([0 5])


%% CV

Y = (19.6412800000000 * xx )*40 + (3 * xx) *2 + (8.05888000000000 *xx ) * 5 * 2 ;
nodes_dist = Y+1 ;


for i=1:length(S1)-1
R1 = S1(1,i+1)-S1(1,i) ;  %ms
R1=R1/1000 ;  %s
CV(1,i) = (nodes_dist*1e-6)/ R1  ;   %m/s
end


%%

figure
plot(s(1,1:length(s)-1) , CV , 'o-m')
xlim([0 50])
title('CV')
xlabel('node #')
ylabel('CV')

%% average CV

mean_CV = mean(CV(1,10:20))
