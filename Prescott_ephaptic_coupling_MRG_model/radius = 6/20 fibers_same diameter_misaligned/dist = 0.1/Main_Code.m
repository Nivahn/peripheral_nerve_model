%% This code is written by Nooshin Abdollahi - Dec7 2021
% Matlab codes necessary for peripheral network modeling. 
% The output of this code will be saved and loaded in Python script.

clc
clear all
close all

%% How many axons of each group?

Desired_Total_Number_Axons = 20   ;     % Total number of axons (both motor and sensory) in a fascicle

Number_Motor_Axons = 0 * Desired_Total_Number_Axons   ;   % Based on literature 15% motor axons

Number_Sensory_Axons =  Desired_Total_Number_Axons - Number_Motor_Axons  ;       % Based on literature 85% sensory axons

Number_C = 0 ;
Number_Adelta = 0 ; 
Number_Abeta = 20 ;

Total_Number_Axons = Number_Motor_Axons + Number_C + Number_Adelta + Number_Abeta  ;   

%% Range of radius (diameter) in each group
% ** C fibers have the the smallest radius

% Motor axons
Motor_Axons_radius = 2.85:8   ;                    % diameter:[5.7 16] => radius:[2.85  8]
mean_Motor_Axons_radius = mean(Motor_Axons_radius) ;  


% Sensory axons
Sensory_C_radius = 0.1:0.1:0.75  ;      % C fiber diameters  [0.2 1.5]
mean_Sensory_C_radius = mean(Sensory_C_radius)   ;

Sensory_Adelta_radius = 0.5:0.1:3  ;     % diameters [1 6]
mean_Sensory_Adelta_radius = mean(Sensory_Adelta_radius) ;

Sensory_Abeta_radius = 3:0.1:6  ;        % diameters [6 12]
mean_Sensory_Abeta_radius = mean(Sensory_Abeta_radius)  ;

Sensory_Aalpha_radius = 6:0.1:10  ;    % I  do not include this  [12 20]
mean_Sensory_Aalpha_radius = mean(Sensory_Aalpha_radius) ;


%%  Distribution of radius (diameters) of different types of axons 
% Distributions are basaed on Gaussian 
% Group1:motor     Group2:C fiber     Group3:Adelta     Group4:Abeta




%%%%%%%%%%%%%%%%%%%%%%%%% Group1: Motor axons %%%%%%%%%%%%%%%%%%%%%%%%%%%%
Motor_Axons_dist = normrnd(mean_Motor_Axons_radius, 0.5  ,[1,Number_Motor_Axons]) ;    % you can change sigma (=0.5).  increasing sigma => wider distribution
for i=1:length(Motor_Axons_dist)
Motor_Axons_dist(1,i) = round(Motor_Axons_dist(1,i),1) ;  % round it to 1 number after decimal
% if Motor_Axons_dist(1,i) < 2.85
%     Motor_Axons_dist(1,i) = 2.85   ;
% end
% 
% if Motor_Axons_dist(1,i) > 8
%     Motor_Axons_dist(1,i) = 8   ;
% end

end
Motor_Axons_dist ;
% figure
% plot(1:Number_Motor_Axons, Motor_Axons_dist , 'ob')
% ylim([0 15])
% title('Motor axons')




%%%%%%%%%%%%%%%%%%%%%%%%%% Sensory axons %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Group2: C fibers
%Sensory_C_dist = abs  (normrnd(mean_Sensory_C_Diam , 0.5  ,[1,Number_C]) ) ; % you can change sigma   
% for i=1:length(Sensory_C_dist)
% Sensory_C_dist(1,i) = round(Sensory_C_dist(1,i),1) ;
% end
Sensory_C_dist = mean_Sensory_C_radius*ones(1,Number_C) ; % all C fibers have the same radius (no heterogenety within C fibers)
Sensory_C_dist ;
figure
plot(1:Number_C, Sensory_C_dist , 'ob')
ylim([0 15])
title('C fibers')
ylabel('radius')
xlabel('#axon')


% Group3: Adelta
Sensory_Adelta_dist = abs  (normrnd(mean_Sensory_Adelta_radius , 0.5  ,[1,Number_Adelta])  )  ;   % you can change sigma,  *abs() becacuse radius values cannot be negative.
for i=1:length(Sensory_Adelta_dist)
Sensory_Adelta_dist(1,i) = round(Sensory_Adelta_dist(1,i),1) ;   % round to have one number after decimal

% if Sensory_Adelta_dist(1,i) < 0.5
%     Sensory_Adelta_dist(1,i) = 0.5   ;
% end
% 
% if Sensory_Adelta_dist(1,i) > 3
%     Sensory_Adelta_dist(1,i) = 3   ;
% end

end
Sensory_Adelta_dist ;
figure
plot(1:Number_Adelta, Sensory_Adelta_dist , 'ob')
ylim([0 15])
title('Adelta fibers')
ylabel('radius')



% Group4: Abeta
Sensory_Abeta_dist = abs  (normrnd(mean_Sensory_Abeta_radius , 0.5  ,[1,Number_Abeta]) ) ;    % you can change the sigma 
for i=1:length(Sensory_Abeta_dist)
Sensory_Abeta_dist(1,i) = 6 ;

% if Sensory_Abeta_dist(1,i) < 3
%     Sensory_Abeta_dist(1,i) = 3   ;
% end
% 
% if Sensory_Abeta_dist(1,i) > 6
%     Sensory_Abeta_dist(1,i) = 6   ;
% end

end
Sensory_Abeta_dist ;
figure
plot(1:Number_Abeta, Sensory_Abeta_dist , 'ob')
ylim([0 15])
title('Abeta fibers')
ylabel('radius')


%%%%%%%%%%%%%%%%%%% put all axons in a vector %%%%%%%%%%%%%%%%%%%%%%%%%%%%
All_Axons(1:Number_Motor_Axons,1) = Motor_Axons_dist ;
All_Axons(Number_Motor_Axons+1:Number_Motor_Axons+Number_C,1) = Sensory_C_dist ;
All_Axons(Number_Motor_Axons+Number_C+1:Number_Motor_Axons+Number_C+Number_Adelta,1) = Sensory_Adelta_dist  ;
All_Axons(Number_Motor_Axons+Number_C+Number_Adelta+1:Number_Motor_Axons+Number_C+Number_Adelta+Number_Abeta,1) = Sensory_Abeta_dist  ;
size(All_Axons);


% which axon belongs to which group?
GG(1:Number_Motor_Axons,1)=1 ;
GG(Number_Motor_Axons+1:Number_Motor_Axons+Number_C,1)=2 ;
GG(Number_Motor_Axons+Number_C+1:Number_Motor_Axons+Number_C+Number_Adelta,1)=3 ;
GG(Number_Motor_Axons+Number_C+Number_Adelta+1:Number_Motor_Axons+Number_C+Number_Adelta+Number_Abeta,1)=4 ;
%I do not save GG, I save G becasue G is the final grouping after shuffling spikes


%%%%%%%%% shuffling the axons in the "All_Axons" vector randomely %%%%%%%%%
% suffling is important for next part (i.e. pack_v)

shuffled_All_Axons = All_Axons(randperm(length(All_Axons)))' ;

for i=1:length(shuffled_All_Axons)
    
    f = find(All_Axons == shuffled_All_Axons(1,i)) ;
    G(i,1)=GG(f(1,1),1) ;
    
end

G ;   %  Grouping for shuffled_All_Axons (save this)

%%  Fiber packing

[C,R] = pack_v(shuffled_All_Axons)   ;    % C:(x,y) (Coordinates)   R includes radius 


%% plotting fiber packing (i.e. location of each axon)

figure
for i=1:length(R)
 
    if G(i,1)==1
        viscircles(C(i,:),R(i,1), 'color', 'b','Linewidth',0.2)    % motor axons are shown in blue
    
    elseif G(i,1)==2
        viscircles(C(i,:),R(i,1), 'color', 'r','Linewidth',0.2)    % C fibers are shown in red
        
    elseif G(i,1)==3
        viscircles(C(i,:),R(i,1), 'color', 'g','Linewidth',0.2)    % Adelta fibers are shown in green
        
    elseif G(i,1)==4
        viscircles(C(i,:),R(i,1), 'color', 'k','Linewidth',0.2)    % Abeta fibers are shown in black
        
    end
     
    
    hold on
end
axis square

%% numbering the axons in the figure above
axis square
a = [0:length(C)-1]'; b = num2str(a); c = cellstr(b);
dx = 0.2; dy = 0.1; % displacement so the text does not overlay the data points
text(C(:,1)-dx, C(:,2), c ,  'FontSize',10);

%%


for i=2:length(R)
    
    R(i,1) = R(i-1,1) + 0.0000001 ;
    
end




%% Next part:  In NEURON, I need to build axons with different diameters. In this section, I will find the morphological parameters (e.g. internodelength)dependent on FiberD  
%I will save the parameters from this section and use it to build the axons in NEURON

unique_radius = unique(R) ;     %unique_radius(1,1): C fiber radius
size(unique_radius)  ;

%%  the parameters (p1,p2,p3) are obtained from another Matlab Code ("MRGcurvefitting")


% figure

for i = 1:length(unique_radius)    % it starts from 2 because C fiber's radius is the smallest one : unique_radius(1,:) and C fibers are unmyelinated
    
x= unique_radius(i,1)*2  ;   % to calculate diameters   
    
p1 =     0.01876  ;
p2 =      0.4787  ;
p3 =      0.1204  ;
axonD = p1*x^2 + p2*x + p3  ;
paraD2 = axonD   ;

p1 =    0.006304  ;
p2 =      0.2071  ;
p3 =      0.5339  ;
nodeD = p1*x^2 + p2*x + p3  ;
paraD1 = nodeD   ;

p1 =     -8.215  ;
p2 =      272.4  ;
p3 =     -780.2  ;

deltax = p1*x^2 + p2*x + p3 ;


p1 =     -0.0199  ;
p2 =       3.016  ;
p3 =       17.44  ;
paralength2 = p1*x^2 + p2*x + p3 ;    % FLUT length

paralength2_segments = 5  ;


p1 =      -0.389 ;
p2 =       14.88  ;
p3 =       9.721  ;
nl = round ( p1*x^2 + p2*x + p3 )  ;


paralength1=3  ;  
nodelength=1.0  ;
interlength=(deltax-nodelength-(2*paralength1)-(2*paralength2)) ;

internode_segments = 40   ;        % number of segments in one internode

parameters(i,1) = axonD  ;   
parameters(i,2) = nodeD ;
parameters(i,3) = deltax ;
parameters(i,4) = paralength2/paralength2_segments          ;
parameters(i,5) = nl ;
parameters(i,6) = interlength/internode_segments            ;      % length of each segment 





% plot(1:6 , parameters(i-1,:) )
% hold on
% title('Parameters')

end


for ww = 2:20

parameters(ww,1) = parameters(1,1)  ;   
parameters(ww,2) = parameters(1,2)  ;
parameters(ww,3) = parameters(1,3)  ;
parameters(ww,4) = parameters(1,4)  ;
parameters(ww,5) = parameters(1,5)  ;
parameters(ww,6) = parameters(1,6)  ;  

end







%%  Next part:   Find the neighbouring axons (for connection)

N=zeros(length(C),length(C)) ;
dist=zeros(length(C),length(C)) ;
Distance_edge=zeros(length(C),length(C)) ;

Distance_Centers = pdist2(C, C)  ;  % distance between centers
 


%%%%%%% Distance between edges of circles not centers of circles %%%%%%%%%

for i=1:length(Distance_Centers)
    for j=1:length(Distance_Centers)
        
        if i ~= j
            Distance_edge(i,j)= Distance_Centers(i,j) - (R(i,1)+R(j,1))   ; 
            
        end
        
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


m= 1000 ;        % minimum distance (threshold) 




for k=1:length(Distance_Centers)
 
for i=1:length(Distance_Centers)
    if k ~= i
       if Distance_Centers(k,i)< m
           N(k,i)=1 ;                  % 1 means connection, 0 means no connection
           dist(k,i) = Distance_Centers(k,i)   ;   % distance between the axons
           
       end
    end
end


    
end


    
for j=1:length(N)
for i=1:length(N)
    if N(j,i)==1 
        N(i,j)=0  ;
        dist(i,j) = 0  ;
    end
end
end




%%   Next part:   Find the sections that will be connected to each other

Total_Length_Axon = 46500   ;      % the total length of axons is "about" ...  (microns)







%%%%%%%%%%%%%%%%%%%%% the other type of axons %%%%%%%%%%%%%%%%%%%%%%%%%%%%

for J=1: length(unique_radius)     
     

    z1=0 ;
    z2=0 ;
    z3=0 ;
    z4=0 ;
    j=1 ;
    i=2 ;
    
    which_radius = J ; 
   
%  
eval(['coordinates_sections_' num2str(J) '(1,1) = 0 ;']); 
eval(['coordinates_sections_' num2str(J) '(1,i) = coordinates_sections_' num2str(J) '(1,i-1)+nodelength;']);
eval(['mid_coordinates_sections_' num2str(J) '(1,j) = 0.5;']);
eval(['name_sections_' num2str(J) '(1,j) = sprintf("node_%d",z1) ;']);
z1=z1+1 ;
i=i+1 ;
j=j+1 ;



eval(['coordinates_sections_' num2str(J) '(1,i) = coordinates_sections_' num2str(J) '(1,i-1)+paralength1;']);
eval(['mid_coordinates_sections_' num2str(J) '(1,j) = (coordinates_sections_' num2str(J) '(1,i) + coordinates_sections_' num2str(J) '(1,i-1) )/2;']);
eval(['name_sections_' num2str(J) '(1,j) = sprintf("MYSA_%d",z2) ;']);
z2=z2+1 ;
i=i+1 ;
j=j+1 ;



    for rr=1:paralength2_segments
    
    eval(['coordinates_sections_' num2str(J) '(1,i) = coordinates_sections_' num2str(J) '(1,i-1)+parameters(which_radius,4);']);
    eval(['mid_coordinates_sections_' num2str(J) '(1,j) = (coordinates_sections_' num2str(J) '(1,i) + coordinates_sections_' num2str(J) '(1,i-1) )/2;']);
    eval(['name_sections_' num2str(J) '(1,j) = sprintf("FLUT_%d",z3)  ;']);
    z3=z3+1 ;
    i=i+1 ;
    j=j+1 ;
    end


      for r=1:internode_segments
          
      eval(['coordinates_sections_' num2str(J) '(1,i) = coordinates_sections_' num2str(J) '(1,i-1)+(parameters(which_radius,6));']);
      eval(['mid_coordinates_sections_' num2str(J) '(1,j) = (coordinates_sections_' num2str(J) '(1,i) + coordinates_sections_' num2str(J) '(1,i-1) )/2;']);
      eval(['name_sections_' num2str(J) '(1,j) = sprintf("STIN_%d",z4) ;']);
      z4=z4+1 ;
      i=i+1 ;
      j=j+1 ;

      end

      
      
    for rr=1:paralength2_segments 
    eval(['coordinates_sections_' num2str(J) '(1,i) = coordinates_sections_' num2str(J) '(1,i-1)+parameters(which_radius,4);']);
    eval(['mid_coordinates_sections_' num2str(J) '(1,j) = (coordinates_sections_' num2str(J) '(1,i) + coordinates_sections_' num2str(J) '(1,i-1) )/2;']);
    eval(['name_sections_' num2str(J) '(1,j) = sprintf("FLUT_%d",z3) ;']);
    z3=z3+1 ;
    i=i+1 ;
    j=j+1 ;
    end


eval(['coordinates_sections_' num2str(J) '(1,i) = coordinates_sections_' num2str(J) '(1,i-1)+paralength1;']);
eval(['mid_coordinates_sections_' num2str(J) '(1,j) = (coordinates_sections_' num2str(J) '(1,i) + coordinates_sections_' num2str(J) '(1,i-1) )/2;']);
eval(['name_sections_' num2str(J) '(1,j) = sprintf("MYSA_%d",z2) ;']);
z2=z2+1 ;
i=i+1 ;
j=j+1 ;



Length_one_set(J,1) = eval(['coordinates_sections_' num2str(J) '(1,i-1)'])  ;  % Length of one set including (node+MYSA+5*FLUT+40*Internodes+5*FLUT+MYSA)
number_sets(J,1) = round (Total_Length_Axon/Length_one_set(J,1))  ;     % how many sets we need for this specific radius in order to get a total length about 10000 microns


%
for ii=2:number_sets(J,1)   % it starts from 2 becasue the first set is already above
    

  
eval(['coordinates_sections_' num2str(J) '(1,i) = coordinates_sections_' num2str(J) '(1,i-1)+nodelength;']);
eval(['mid_coordinates_sections_' num2str(J) '(1,j) =  (coordinates_sections_' num2str(J) '(1,i) + coordinates_sections_' num2str(J) '(1,i-1) )/2;']);
eval(['name_sections_' num2str(J) '(1,j) = sprintf("node_%d",z1) ;']);
z1=z1+1 ;
i=i+1 ;
j=j+1 ;



eval(['coordinates_sections_' num2str(J) '(1,i) = coordinates_sections_' num2str(J) '(1,i-1)+paralength1;']);
eval(['mid_coordinates_sections_' num2str(J) '(1,j) = (coordinates_sections_' num2str(J) '(1,i) + coordinates_sections_' num2str(J) '(1,i-1) )/2;']);
eval(['name_sections_' num2str(J) '(1,j) = sprintf("MYSA_%d",z2) ;']);
z2=z2+1 ;
i=i+1 ;
j=j+1 ;


    for rr=1:paralength2_segments
    eval(['coordinates_sections_' num2str(J) '(1,i) = coordinates_sections_' num2str(J) '(1,i-1)+parameters(which_radius,4);']);
    eval(['mid_coordinates_sections_' num2str(J) '(1,j) = (coordinates_sections_' num2str(J) '(1,i) + coordinates_sections_' num2str(J) '(1,i-1) )/2;']);
    eval(['name_sections_' num2str(J) '(1,j) = sprintf("FLUT_%d",z3) ;']);
    z3=z3+1 ;
    i=i+1 ;
    j=j+1 ;
    end


      for r=1:internode_segments
          
      eval(['coordinates_sections_' num2str(J) '(1,i) = coordinates_sections_' num2str(J) '(1,i-1)+(parameters(which_radius,6));']);
      eval(['mid_coordinates_sections_' num2str(J) '(1,j) = (coordinates_sections_' num2str(J) '(1,i) + coordinates_sections_' num2str(J) '(1,i-1) )/2;']);
      eval(['name_sections_' num2str(J) '(1,j) = sprintf("STIN_%d",z4) ;']);
      z4=z4+1 ;
      i=i+1 ;
      j=j+1 ;

      end
      
      
      
    for rr=1:paralength2_segments 
    eval(['coordinates_sections_' num2str(J) '(1,i) = coordinates_sections_' num2str(J) '(1,i-1)+parameters(which_radius,4);']);
    eval(['mid_coordinates_sections_' num2str(J) '(1,j) = (coordinates_sections_' num2str(J) '(1,i) + coordinates_sections_' num2str(J) '(1,i-1) )/2;']);
    eval(['name_sections_' num2str(J) '(1,j) = sprintf("FLUT_%d",z3) ;']);
    z3=z3+1 ;
    i=i+1 ;
    j=j+1 ;
    end

    
    

eval(['coordinates_sections_' num2str(J) '(1,i) = coordinates_sections_' num2str(J) '(1,i-1)+paralength1;']);
eval(['mid_coordinates_sections_' num2str(J) '(1,j) = (coordinates_sections_' num2str(J) '(1,i) + coordinates_sections_' num2str(J) '(1,i-1) )/2;']);
eval(['name_sections_' num2str(J) '(1,j) = sprintf("MYSA_%d",z2) ;']);
z2=z2+1 ;
i=i+1 ;
j=j+1 ;


    
end

Length_All_sets(J,1) = eval(['coordinates_sections_' num2str(J) '(1,i-1)'])  ;   % "around" 10000 microns
Number_of_nodes (J,1) = z1 ;



end


%%

% a = 0  ;
% b= 208 ;
% 
% rrr = (b-a).*rand(20,1) + a ;


rrr = [67.4678402083890 ;
330.422368924505;
129.465457490639;
219.869784370584;
68.9098714719089;
250.424487623081;
109.396054368700;
272.096904966341;
286.713233306243;
311.231062614663;
187.425304977039;
34.8696932467240;
95.2544189861966;
379.948342384695;
63.3892558911968;
343.539862635652;
223.950453068184;
414.392042116784;
32.5210199613244;
184.154160226586] ;





%%

coordinates_sections_2 = coordinates_sections_2 - rrr(1,1)  ;       
mid_coordinates_sections_2 = mid_coordinates_sections_2 - rrr(1,1)  ; 

coordinates_sections_3 = coordinates_sections_3 - rrr(2,1)  ;       
mid_coordinates_sections_3 = mid_coordinates_sections_3 - rrr(2,1)  ; 

coordinates_sections_4 = coordinates_sections_4 - rrr(3,1)  ;       
mid_coordinates_sections_4 = mid_coordinates_sections_4 - rrr(3,1)  ; 

coordinates_sections_5 = coordinates_sections_5 - rrr(4,1)  ;       
mid_coordinates_sections_5 = mid_coordinates_sections_5 - rrr(4,1)  ; 

coordinates_sections_6 = coordinates_sections_6 - rrr(5,1)  ;       
mid_coordinates_sections_6 = mid_coordinates_sections_6 - rrr(5,1)  ; 

coordinates_sections_7 = coordinates_sections_7 - rrr(6,1)  ;       
mid_coordinates_sections_7 = mid_coordinates_sections_7 - rrr(6,1)  ; 

coordinates_sections_8 = coordinates_sections_8 - rrr(7,1)  ;       
mid_coordinates_sections_8 = mid_coordinates_sections_8 - rrr(7,1)  ; 

coordinates_sections_9 = coordinates_sections_9 - rrr(8,1)  ;       
mid_coordinates_sections_9 = mid_coordinates_sections_9 - rrr(8,1)  ; 


coordinates_sections_10 = coordinates_sections_10 - rrr(9,1)  ;       
mid_coordinates_sections_10 = mid_coordinates_sections_10 - rrr(9,1)  ; 


coordinates_sections_11 = coordinates_sections_11 - rrr(10,1)  ;       
mid_coordinates_sections_11 = mid_coordinates_sections_11 - rrr(10,1)  ; 


coordinates_sections_12 = coordinates_sections_12 - rrr(11,1)  ;       
mid_coordinates_sections_12 = mid_coordinates_sections_12 - rrr(11,1)  ; 


coordinates_sections_13 = coordinates_sections_13 - rrr(12,1)  ;       
mid_coordinates_sections_13 = mid_coordinates_sections_13 - rrr(12,1)  ; 

coordinates_sections_14 = coordinates_sections_14 - rrr(13,1)  ;       
mid_coordinates_sections_14 = mid_coordinates_sections_14 - rrr(13,1)  ; 

coordinates_sections_15 = coordinates_sections_15 - rrr(14,1)  ;       
mid_coordinates_sections_15 = mid_coordinates_sections_15 - rrr(14,1)  ; 

coordinates_sections_16 = coordinates_sections_16 - rrr(15,1)  ;       
mid_coordinates_sections_16 = mid_coordinates_sections_16 - rrr(15,1)  ; 


coordinates_sections_17 = coordinates_sections_17 - rrr(16,1)  ;       
mid_coordinates_sections_17 = mid_coordinates_sections_17 - rrr(16,1)  ; 


coordinates_sections_18 = coordinates_sections_18 - rrr(17,1)  ;       
mid_coordinates_sections_18 = mid_coordinates_sections_18 - rrr(17,1)  ; 


coordinates_sections_19 = coordinates_sections_19 - rrr(18,1)  ;       
mid_coordinates_sections_19 = mid_coordinates_sections_19 - rrr(18,1)  ; 


coordinates_sections_20 = coordinates_sections_20 - rrr(19,1)  ;       
mid_coordinates_sections_20 = mid_coordinates_sections_20 - rrr(19,1)  ; 


%%

% 
figure
plot(coordinates_sections_3 ,1 , 'ob')
hold on
plot(coordinates_sections_6 ,1.05 , 'or')   
ylim([1 1.5])
%xlim([0 10000])




%%

compartments_in_a_set = 3 + internode_segments + (2*paralength2_segments) ;

 
 
%%




%%%%%%%%%% Connect other axons (excluding C fibers) to axons %%%%%%%%%%%%%

 for s=1 :length((unique_radius))   
     for t=1 :length((unique_radius))
         
         if t> s
         
             
             clear S_first
             clear S_second
             clear n nn f1 f2 fz m mm
             L1=0 ;
             L2=0 ;
             p=1 ; 
            
             if eval(['coordinates_sections_' num2str(t) '(1,1)']) == eval(['coordinates_sections_' num2str(s) '(1,1)'])  
              S_first(p,1) =  sprintf("node_%d",L1)  ;
              S_second(p,1) = sprintf("node_%d",L2)  ;
              
              p=p+1 ;
              L1=L1+1 ;
              L2=L2+1 ;
              
             end

              
              
              
             for n2=((L2*compartments_in_a_set)+1):compartments_in_a_set:compartments_in_a_set*Number_of_nodes(t,1) 
                 f2 = eval(['mid_coordinates_sections_' num2str(t) '(1,n2)'])   ;
                 
                 for n1=((L1*compartments_in_a_set)+1):compartments_in_a_set:compartments_in_a_set*Number_of_nodes(s,1) 
                 
                     f1 = eval(['mid_coordinates_sections_' num2str(s) '(1,n1)'])   ;
                 
                     if f1 < f2
                          for m=1:length (eval(['coordinates_sections_' num2str(t)]))-1
                             if f1 >=  eval(['coordinates_sections_' num2str(t) '(1,m)']) && f1 <= eval(['coordinates_sections_' num2str(t) '(1,m+1)'])
                                 S_first(p,1) = sprintf("node_%d",L1)  ;
                                 S_second(p,1) = eval(['name_sections_' num2str(t) '(1,m)'])  ;
                                 p=p+1 ;
                             end
                          end
                          L1=L1+1 ;

                     end
                 end
                 
                 
                  for mm=1:length (eval(['coordinates_sections_' num2str(s)]))-1
                         if f2 >=  eval(['coordinates_sections_' num2str(s) '(1,mm)']) && f2 <= eval(['coordinates_sections_' num2str(s) '(1,mm+1)'])
                             S_first(p,1) =  eval(['name_sections_' num2str(s) '(1,mm)'])  ;
                             S_second(p,1) = sprintf("node_%d",L2)  ;
                             p=p+1 ;
                         end
                  end
                  L2=L2+1  ;
 
             end
             
             
                     
             if  Number_of_nodes(s,1) > Number_of_nodes(t,1)  
                 
                 for nn=((L1*compartments_in_a_set)+1):compartments_in_a_set: compartments_in_a_set*Number_of_nodes(s,1)
                     fz=eval(['mid_coordinates_sections_' num2str(s) '(1,nn)'])  ; 
                     for m=1:length (eval(['coordinates_sections_' num2str(t)]))-1
                         if fz >=  eval(['coordinates_sections_' num2str(t) '(1,m)']) && fz <= eval(['coordinates_sections_' num2str(t) '(1,m+1)']) 
                             S_first(p,1) = sprintf("node_%d",L1)  ;
                             S_second(p,1) = eval(['name_sections_' num2str(t) '(1,m)'])  ;
                             p=p+1 ;
                         end
                         
                     end
                     L1=L1+1 ;
                 end
             end   
                 
             
             
             
             
             
               
             
             
             eval(['Connect_types_' num2str(s),num2str(t) '(1:length(S_first),1) = S_first ;']);
             eval(['Connect_types_' num2str(s),num2str(t) '(length(S_first)+1:length(S_first)+length(S_second),1) = S_second ;']);
            
            %%%%%%%%%% save data %%%%%%%%%%
            clear SAVE
            SAVE = char( eval(['Connect_types_' num2str(s),num2str(t)])) ;
            filename=['Connect_types_' num2str(s),num2str(t) '.mat']; 
            save(filename , 'SAVE');    
             
             
         end


     end
 end


%%
%%%%%%%%%%%%%%%%%%%%% Connect each axon to itself %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 
 for s=1 :length((unique_radius))   
     for t=1 :length((unique_radius))
         
         if t== s
         
             
             clear S_first
             clear S_second
             clear n nn f1 f2 fz m mm
             L1=0 ;
             L2=0 ;
             p=1 ; 
            
             
             for n=0:Number_of_nodes(s,1)-1 
             
              S_first(p,1) =  sprintf("node_%d",n)  ;
              S_second(p,1) = sprintf("node_%d",n)  ;
              p=p+1 ;

             end
              

            eval(['Connect_types_' num2str(s),num2str(t) '(1:length(S_first),1) = S_first ;']);
            eval(['Connect_types_' num2str(s),num2str(t) '(length(S_first)+1:length(S_first)+length(S_second),1) = S_second ;']);
            
            %%%%%%%%%% save data %%%%%%%%%%
            clear SAVE
            SAVE = char( eval(['Connect_types_' num2str(s),num2str(t)])) ;
            filename=['Connect_types_' num2str(s),num2str(t) '.mat']; 
            save(filename , 'SAVE');    
             
             
         end


     end
 end 
 
 

 
 






 %% Output 
% *** Connection_types_xx  are saved above


save('R.txt' , 'R' , '-ascii')   % save radius of all axons
save('C.txt' , 'C' , '-ascii')   % save location of all axons (x,y)
save('G.txt' , 'G' , '-ascii')   % save Groups of all axons. i.e., each radius in "R" belongs to which type (Group1,2,3 or 4)             

% To be used in NEURON to build models
save('parameters.txt' , 'parameters' , '-ascii')
save('Number_of_nodes.txt' , 'Number_of_nodes' , '-ascii')



save('unique_radius.txt' , 'unique_radius' , '-ascii')
save('neighboringAxon.txt' , 'N' , '-ascii')
% % save('neighbouringAxon.mat','N')

save('dist.txt' , 'dist' , '-ascii')

%
save('Distance_edge.txt' , 'Distance_edge' , '-ascii')





%% Next:  Boundary Condition

% Boundary_coordinates = zeros(20,2) ;
% 
% 
% %%
% x=1 ;
% for i=1:20 
%     
%     Boundary_coordinates(i,2) = 40 ;
%     Boundary_coordinates(i,1) = x ;
%     x=x+1 ;
%     
% end
% 
% 
% %%
% v = ones(1,20) ;
% Boundary_neighboring = diag(v) ;
% 
% Boundary_dist = Boundary_neighboring ;
% 
% Boundary_dist(0+1,0+1) = Boundary_neighboring(0+1,0+1)*(8+8+0.1+0.1) ;
% 
% 
% Boundary_dist(1+1,1+1) = Boundary_neighboring(1+1,1+1)*(8+0.1) ;
% Boundary_dist(2+1,2+1) = Boundary_neighboring(2+1,2+1)*(8+0.1) ;
% Boundary_dist(3+1,3+1) = Boundary_neighboring(3+1,3+1)*(8+0.1) ;
% Boundary_dist(9+1,9+1) = Boundary_neighboring(9+1,9+1)*(8+0.1) ;
% Boundary_dist(11+1,11+1) = Boundary_neighboring(11+1,11+1)*(8+0.1) ;
% Boundary_dist(12+1,12+1) = Boundary_neighboring(12+1,12+1)*(8+0.1) ;

%%
% 
% % finding the neurons on the edge, for now I do it manually
% 
% neurons_on_edge_up = [10 8 4    ] ;
% neurons_on_edge_right = [7 5 6 13    ] ;
% neurons_on_edge_down = [14 18 17  ] ;
% neurons_on_edge_left = [19 15 16  ] ;
% 
% neurons_on_edge = [neurons_on_edge_up, neurons_on_edge_right, neurons_on_edge_down, neurons_on_edge_left] ; 
% 
% % finding the boundary cable coordinates
% distance_boundary_to_neuron = 7 ;   % microns
% 
% for i=1:length(neurons_on_edge_up)
%     B_coordinates_up(i,1) = C(neurons_on_edge_up(i)+1,1) ;
%     B_coordinates_up(i,2) = C(neurons_on_edge_up(i)+1,2) + distance_boundary_to_neuron ; 
% end
% 
% for i=1:length(neurons_on_edge_right)
%     B_coordinates_right(i,1) = C(neurons_on_edge_right(i)+1,1) + distance_boundary_to_neuron ;
%     B_coordinates_right(i,2) = C(neurons_on_edge_right(i)+1,2)  ; 
% end
% 
% for i=1:length(neurons_on_edge_down)
%     B_coordinates_down(i,1) = C(neurons_on_edge_down(i)+1,1)  ;
%     B_coordinates_down(i,2) = C(neurons_on_edge_down(i)+1,2) - distance_boundary_to_neuron ; 
% end
%     
% for i=1:length(neurons_on_edge_left)
%     B_coordinates_left(i,1) = C(neurons_on_edge_left(i)+1,1) - distance_boundary_to_neuron ;
%     B_coordinates_left(i,2) = C(neurons_on_edge_left(i)+1,2)  ; 
% end    
%     
% Boundary_coordinates(:,1) = [B_coordinates_up(:,1)', B_coordinates_right(:,1)', B_coordinates_down(:,1)', B_coordinates_left(:,1)'] ;
% Boundary_coordinates(:,2) = [B_coordinates_up(:,2)', B_coordinates_right(:,2)', B_coordinates_down(:,2)', B_coordinates_left(:,2)'] ;
% Boundary_coordinates ;
% 
% plot(Boundary_coordinates(:,1),Boundary_coordinates(:,2),'o','Color',[0.9290 0.6940 0.1250])
% 
% %% finding the neighboring neurons to which boundary cables are connected 
% 
% C_and_Boundary_coordinates(:,1) = [ C(:,1)' , Boundary_coordinates(:,1)' ]  ;
% C_and_Boundary_coordinates(:,2) = [ C(:,2)' , Boundary_coordinates(:,2)' ]  ;
% 
% size_C = size(C) ;
% size_Boundary_coordinates = size(Boundary_coordinates) ;
% 
% Boundary_N=zeros(length(C_and_Boundary_coordinates),length(C_and_Boundary_coordinates)) ;
% Boundary_dist=zeros(length(C_and_Boundary_coordinates),length(C_and_Boundary_coordinates)) ;
% Boundary_Distance = pdist2(C_and_Boundary_coordinates, C_and_Boundary_coordinates)  ;
% 
% 
% 
% m= 8 ;        % minimum distance (threshold)
% 
% 
% 
% 
% for k=1:length(Boundary_Distance)
% for i=1:length(Boundary_Distance)
%     if k ~= i
%        if Boundary_Distance(k,i)< m
%            Boundary_N(k,i)=1 ;
%            Boundary_dist(k,i)=Boundary_Distance(k,i)  ;
%           
%        end
%     end
% end
%    
% end
% 
% Boundary_neighboring = Boundary_N(size_C+1:size(Boundary_N) ,1:size_C) ;
% Boundary_neighboring = Boundary_neighboring.*0 ;
% 
% Boundary_neighboring(1,10+1) = 1 ;
% Boundary_neighboring(2,8+1) = 1 ;
% Boundary_neighboring(3,4+1) = 1 ;
% Boundary_neighboring(4,7+1) = 1 ;
% Boundary_neighboring(5,5+1) = 1 ;
% Boundary_neighboring(6,6+1) = 1 ;
% Boundary_neighboring(7,13+1) = 1 ;
% Boundary_neighboring(8,14+1) = 1 ;
% Boundary_neighboring(9,18+1) = 1 ;
% Boundary_neighboring(10,17+1) = 1 ;
% Boundary_neighboring(11,19+1) = 1 ;
% Boundary_neighboring(12,15+1) = 1 ;
% Boundary_neighboring(13,16+1) = 1 ;
% 
% 
% Boundary_dist = Boundary_dist(size_C+1:size(Boundary_N) ,1:size_C) ;
% Boundary_dist = Boundary_neighboring.*7 ;
% 
% 
% 
% 




%%



%%
% Boundary_neighboring(2,13)= 0 ;
% Boundary_dist(2,13) = 0 ;
% 
% Boundary_neighboring(7,16)= 0 ;
% Boundary_dist(7,16) = 0 ;



%% find the sections for connections


% for Boundary Cable
  
z=0 ;
i=2 ;
j=1 ;


number_of_sections = 1000  ;
sectionlength= Total_Length_Axon/number_of_sections  ;      



for ii=0:number_of_sections-1
    
Boundary_coordinates_sections(1,1) = 0  ;
Boundary_coordinates_sections(1,i) = Boundary_coordinates_sections(1,i-1) + sectionlength  ;
Boundary_mid_coordinates_section(1,j) = (Boundary_coordinates_sections(1,i) + Boundary_coordinates_sections(1,i-1) )/2  ;
Boundary_name_sections(1,j) = sprintf("section_%d",z)  ;
i=i+1 ;
z=z+1 ;
j=j+1 ;

end




for t=1 :length((unique_radius))
         

             clear S_first
             clear S_second
             clear n f1 m 
             L=0 ;
             p=1 ; 
             
             for n=1:compartments_in_a_set:compartments_in_a_set*Number_of_nodes(t,1)
                 
                 f1 = eval(['mid_coordinates_sections_' num2str(t) '(1,n)'])  ; 
    
                     for m=1:length (Boundary_coordinates_sections)-1
                         if f1 >= Boundary_coordinates_sections(1,m)  && f1 <= Boundary_coordinates_sections(1,m+1)
                             S_first(p,1) = sprintf("node_%d",L)  ;
                             S_second(p,1) = Boundary_name_sections(1,m)  ;
                             p=p+1 ;
                         end
                     end
                     

                     L=L+1 ;
                
    
             end
 
            eval(['Boundary_to_' num2str(t) '(1:length(S_first),1) = S_first ;']);
            eval(['Boundary_to_' num2str(t) '(length(S_first)+1:length(S_first)+length(S_second),1) = S_second ;']);
            
            %%%%%%%%%% save data %%%%%%%%%%
            clear SAVE
            SAVE = char( eval(['Boundary_to_' num2str(t)])) ;
            filename=['Boundary_to_' num2str(t) '.mat']; 
            save(filename , 'SAVE');

end
     


%%% Boundary to Boundary are not connected




%% Output 
% *** Boundary_to_x are saved above


save('Boundary_coordinates.txt' , 'Boundary_coordinates' , '-ascii')    % (x,y)
save('Boundary_neighboring.txt' , 'Boundary_neighboring' , '-ascii')     
save('Boundary_dist.txt' , 'Boundary_dist' , '-ascii') 











%%   Next step:    Transverse resistances (g)  unit: S/cm2


% Calculating c in PLOS computational biology paper, Equation 9


for i=1 :length((unique_radius))    
    for j=1  :length((unique_radius))
        
        
        if j>i
            
        clear Rg
      
        z=2 ;
        
        L = length(eval(['Connect_types_' num2str(i),num2str(j)]))/2    ;
        
        Rg=zeros(L,1)  ;

        for Z=2:L-1


           CO1 = find( eval(['name_sections_' num2str(i)]) == eval(['Connect_types_' num2str(i),num2str(j) '(Z-1,1)'])  );
           CO2 = find( eval(['name_sections_' num2str(i)]) == eval(['Connect_types_' num2str(i),num2str(j) '(Z+1,1)']));
           g1 = eval(['mid_coordinates_sections_' num2str(i) '(1,CO1)'])   ;
           g2 = eval(['mid_coordinates_sections_' num2str(i) '(1,CO2)'])   ;
           
           Rg(z,1) = (g2-g1)/2  ;
           z=z+1 ;
           
           
  
           if g2 < g1
                 msg = 'Error occurred. g2<g1';
                 error(msg)
           end

        end
        
        
           CO1 = find( eval(['name_sections_' num2str(i)]) == eval(['Connect_types_' num2str(i),num2str(j) '(1,1)'])  );
           CO2 = find( eval(['name_sections_' num2str(i)]) == eval(['Connect_types_' num2str(i),num2str(j) '(2,1)']));
           g1 = eval(['mid_coordinates_sections_' num2str(i) '(1,CO1)'])   ;
           g2 = eval(['mid_coordinates_sections_' num2str(i) '(1,CO2)'])   ;
           Rg(1,1) = (g2-g1)/2  ;
           
           CO1 = find( eval(['name_sections_' num2str(i)]) == eval(['Connect_types_' num2str(i),num2str(j) '(L-1,1)'])  );
           CO2 = find( eval(['name_sections_' num2str(i)]) == eval(['Connect_types_' num2str(i),num2str(j) '(L,1)']));
           g1 = eval(['mid_coordinates_sections_' num2str(i) '(1,CO1)'])   ;
           g2 = eval(['mid_coordinates_sections_' num2str(i) '(1,CO2)'])   ;
           Rg(L,1) = (g2-g1)/2  ;
   
        
        
            %%%%%%%%%% save data %%%%%%%%%%
            filename=['Rg_' num2str(i),num2str(j) '.txt']; 
            save(filename ,  'Rg' , '-ascii') 

        
        

        end

    end
end


%%


%%%  Areas of the nodes 

for i=1 :length((unique_radius))
    for j=2  :length((unique_radius))
        
        if j>i
            
            clear Areas

            k=0 ;
 
            L = length(eval(['Connect_types_' num2str(i),num2str(j)]))/2    ;
            Areas =zeros(L,1)  ;

            for Z=1:L

                   if (eval(['Connect_types_' num2str(i),num2str(j) '(Z,1)']) == sprintf("node_%d",k) )  
                       Areas(Z,1) = pi*parameters(i,2)  ;   % surface area = pi*d*L  (L=1),   parameters(i,2): nodeD
                       
                       k=k+1 ;
                       
                   else
                       
                       Areas(Z,1) = pi*parameters(j,2)  ;   % surface area = pi*d*L  (L=1),   parameters(i,2): nodeD

                   end  
            end
            
            
            %%%%%%%%%% save data %%%%%%%%%%

            filename=['Areas_' num2str(i),num2str(j) '.txt']; 
            save(filename ,  'Areas' , '-ascii') 

            
            
            
            
        end
    end
end






%% between boundary cables and the axons (C fibers and other axons) %%%%%



 for j=1:length((unique_radius))
        
        clear Rg
      
        z=2 ;
        
        L = length(eval(['Boundary_to_' num2str(j)]))/2    ;

        Rg=zeros(L,1)  ;
        


        for Z=2:L-1
            
           CO1 = find( eval(['name_sections_' num2str(j)]) == eval(['Boundary_to_' num2str(j) '(Z-1,1)'])  );
           CO2 = find( eval(['name_sections_' num2str(j)]) == eval(['Boundary_to_' num2str(j) '(Z+1,1)'])  );
           g1 = eval(['mid_coordinates_sections_' num2str(j) '(1,CO1)'])   ;
           g2 = eval(['mid_coordinates_sections_' num2str(j) '(1,CO2)'])   ;
           
           Rg(z,1) = (g2-g1)/2  ;
           z=z+1 ;
           
           
        end
           
           
           CO1 = find( eval(['name_sections_' num2str(j)]) == eval(['Boundary_to_' num2str(j) '(1,1)'])  );
           CO2 = find( eval(['name_sections_' num2str(j)]) == eval(['Boundary_to_' num2str(j) '(2,1)']));
           g1 = eval(['mid_coordinates_sections_' num2str(j) '(1,CO1)'])   ;
           g2 = eval(['mid_coordinates_sections_' num2str(j) '(1,CO2)'])   ;
           Rg(1,1) = (g2-g1)/2  ;
           
           CO1 = find( eval(['name_sections_' num2str(j)]) == eval(['Boundary_to_' num2str(j) '(L-1,1)'])  );
           CO2 = find( eval(['name_sections_' num2str(j)]) == eval(['Boundary_to_' num2str(j) '(L,1)']));
           g1 = eval(['mid_coordinates_sections_' num2str(j) '(1,CO1)'])   ;
           g2 = eval(['mid_coordinates_sections_' num2str(j) '(1,CO2)'])   ;
           Rg(L,1) = (g2-g1)/2  ;

        
        
            %%%%%%%%%% save data %%%%%%%%%%
            filename=['Boundary_Rg_' num2str(j) '.txt']; 
            save(filename ,  'Rg' , '-ascii') 
           
           
 
 end



%%




















%%
% 
% figure
% 
% for i=1:length(C)
%     
%     plot(C(i,1),C(i,2),'ob')
%     hold on
%     
%     
% end
% 
% for i=1:length(Boundary_coordinates)
%     
%     plot(Boundary_coordinates(i,1),Boundary_coordinates(i,2),'or')
%     hold on
%     
%     
% end
















