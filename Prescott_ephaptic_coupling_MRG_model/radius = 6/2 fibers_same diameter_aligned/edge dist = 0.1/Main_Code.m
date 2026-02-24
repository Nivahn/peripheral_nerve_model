%% This code is written by Nooshin Abdollahi - Oct 23 2022
% Matlab codes necessary for peripheral network modeling. 
% The output of this code will be saved and loaded in Python script.

clc
clear all
close all



%%

R = [6;6]  ; % two fibers with exactly the same radius. 
C(1,1) = 0 ;
C(1,2) = 0 ;

C(2,1) = 12.1 ;   % the edge to edge distance between the two fibers is 4 microns. (12-4*2) = 4  
C(2,2) = 0 ;

G(1,1) = 4 ;  % Both fibers are Abeta fibers (group 4) ;
G(1,2) = 4 ; 


figure
for i=1:length(R)
 
        viscircles(C(i,:),R(i,1), 'color', 'k','Linewidth',0.2)    
        hold on
end

% xlim([-20 20])
% ylim([-20 20])
axis equal

%%

unique_radius = unique(R) ;     


%%  To find morphological parameters 


i = 1    ;
    
x= unique_radius(i,1)*2  ;   % fiber diameter
    
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




%%  

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
%%


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




%%   

Total_Length_Axon = 46500   ;      % the total length of axons is "about" ...  (microns)



%%%%%%%%%%%%%%%%%%%%% the other type of axons %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% there is only one type 

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

figure
plot(coordinates_sections_1 ,1 , 'ob')
hold on
plot(coordinates_sections_1 ,1.05 , 'or')

ylim([1 1.5])



%%

compartments_in_a_set = 3 + internode_segments + (2*paralength2_segments) ;

 

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


save('Distance_edge.txt' , 'Distance_edge' , '-ascii')





%% Next:  Boundary Condition


% finding the neurons on the edge, for now I do it manually

neurons_on_edge_up = [0 1    ] ;
% neurons_on_edge_right = [    ] ;
% neurons_on_edge_down = [ ] ;
% neurons_on_edge_left = [ ] ;

neurons_on_edge = [neurons_on_edge_up]     ;   

% finding the boundary cable coordinates
distance_boundary_to_neuron = 7 ;   % microns

for i=1:length(neurons_on_edge_up)
    B_coordinates_up(i,1) = C(neurons_on_edge_up(i)+1,1) ;
    B_coordinates_up(i,2) = C(neurons_on_edge_up(i)+1,2) + distance_boundary_to_neuron ; 
end


    
Boundary_coordinates(:,1) = [B_coordinates_up(:,1)']   ;
Boundary_coordinates(:,2) = [B_coordinates_up(:,2)']   ;
Boundary_coordinates ;

plot(Boundary_coordinates(:,1),Boundary_coordinates(:,2),'o','Color',[0.9290 0.6940 0.1250])

%% finding the neighboring neurons to which boundary cables are connected 

C_and_Boundary_coordinates(:,1) = [ C(:,1)' , Boundary_coordinates(:,1)' ]  ;
C_and_Boundary_coordinates(:,2) = [ C(:,2)' , Boundary_coordinates(:,2)' ]  ;

size_C = size(C) ;
size_Boundary_coordinates = size(Boundary_coordinates) ;

Boundary_N=zeros(length(C_and_Boundary_coordinates),length(C_and_Boundary_coordinates)) ;
Boundary_dist=zeros(length(C_and_Boundary_coordinates),length(C_and_Boundary_coordinates)) ;
Boundary_Distance = pdist2(C_and_Boundary_coordinates, C_and_Boundary_coordinates)  ;



m= 8 ;        % minimum distance (threshold)




for k=1:length(Boundary_Distance)
for i=1:length(Boundary_Distance)
    if k ~= i
       if Boundary_Distance(k,i)< m
           Boundary_N(k,i)=1 ;
           Boundary_dist(k,i)=Boundary_Distance(k,i)  ;
          
       end
    end
end
   
end

Boundary_neighboring = Boundary_N(size_C+1:size(Boundary_N) ,1:size_C) ;
Boundary_dist = Boundary_dist(size_C+1:size(Boundary_N) ,1:size_C) ;




%% find the sections for connections


% for Boundary Cable
  
z=0 ;
i=2 ;
j=1 ;


number_of_sections = 4000  ;
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
        
        
        if j>=i
            
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
                       Areas(Z,1) = pi*parameters(i+1,2)  ;   % surface area = pi*d*L  (L=1),   parameters(i,2): nodeD
                       
                       k=k+1 ;
                       
                   else
                       
                       Areas(Z,1) = pi*parameters(j+1,2)  ;   % surface area = pi*d*L  (L=1),   parameters(i,2): nodeD

                   end  
            end
            
            
            %%%%%%%%%% save data %%%%%%%%%%

            filename=['Areas_' num2str(i),num2str(j) '.txt']; 
            save(filename ,  'Areas' , '-ascii') 

            
            
            
            
        end
    end
end







%% between boundary cables and the axons %%%%%



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
b=1 ;

        for Z=1:(length(Connect_types_11)/2)-1


           COO1 = find( eval(['name_sections_' num2str(1)]) == eval(['Connect_types_' num2str(1),num2str(1) '(Z+1,1)'])  );
           COO2 = find( eval(['name_sections_' num2str(1)]) == eval(['Connect_types_' num2str(1),num2str(1) '(Z,1)']));
           gg1 = eval(['mid_coordinates_sections_' num2str(i) '(1,COO1)'])   ;
           gg2 = eval(['mid_coordinates_sections_' num2str(i) '(1,COO2)'])   ;
           
           nodes_dist(b,1) = gg1-gg2 ;
           b=b+1 ;


        end

