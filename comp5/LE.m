% Load image Training dataset (10 32x32 colour images in 40 classes)
sigma=0.1;
 TrainSetCategorical=imageSet('data','recursive');
 XTrain=zeros(112 ,92, 200) ;
 k=1;
 trainNames = {TrainSetCategorical.Description};
 Labels=cell(180,1);
 Testlabels=cell(20,1);
 for i=1:20
     for j=1:9
         Labels{j+9*(i-1)}=trainNames{i};
     end
      Testlabels{i}=trainNames{i};
 end
 
 text=["1","2","3","4","5","6","7","8","9","10"];
 
for i=1:length(TrainSetCategorical)/2
    for j=1:TrainSetCategorical(i).Count  
        I=read(TrainSetCategorical(i),j);
        %// Adjust intensities in image I to range from 0 to 1
        I = I - min(I(:));
        I = I / max(I(:));
        XTrain(:,:, k)=I;
        k=k+1;
    end
end


eigvalues50=cell(10,1);
eigvector50=cell(10,1);
eigvalues100=cell(10,1);
eigvector100=cell(10,1);
stop=1;
y50=cell(1, 10);
y100=cell(1, 10);
label_test_data50=cell(1, 10);
label_test_data100=cell(1, 10);
Mdl50=cell(10,1);
Mdl100=cell(10,1);
cm50=cell(10,1);
cm100=cell(10,1);
for t=1:10

Wieghts=zeros(180, 180);
K=zeros(180,180);
index=zeros(10,1);
TestIm=zeros(112 ,92, 20);
    
 for l=1:20
 index(l)=randi([1+10*(l-1) 10+10*(l-1)]);
 TestIm(:,:,l)=XTrain(:,:,index(l));
 end
 p=1;
XTrainNew=zeros(112 ,92, 180) ;  
for i=1:size(XTrain, 3)
    if ~ismember(i, index)
         XTrainNew(:,:,p)=XTrain(:,:,i);
         p=p+1;
     end
end

for i=1:size(XTrainNew, 3)
    for j=1:size(XTrainNew,3)
       
        K(i,j)=exp(-((norm(XTrainNew(:,:,i)-XTrainNew(:,:,j)))^2)/2*sigma^2);
        if K(i,j)<sigma
        K(i,j)=0;
        else
        K(i,j)=1;    
        end
        
    end
end

col=9*ones(20,1);
row=9*ones(20,1);

Knew= mat2cell(K, col, row );
for q=1:20
    for i=1:9
        for j=1:9
        Wieghts(i+9*(q-1),j+9*(q-1))=Knew{q,q}(i,j);
    
        end
    end
end

%Diagnol matrix
D=zeros(180,180);
sum=0;
for i=1:size(Wieghts, 1)
    for j=1:size(Wieghts,2)
       
        sum=sum+Wieghts(i,j);
         
    end
    D(i,i)=sum;
    sum=0;
end

L=D-Wieghts;
r = 112;
c = 92;
L0=inv(D)*L;

[v,d] = eigs(L0,50,'sa');
eigvalues50{t}=diag(real(d));
eigvector50{t}=real(v);

x=zeros(size(eigvector50{t},1)*size(eigvector50{t},2),1);
y=zeros(size(eigvector50{t},1)*size(eigvector50{t},2),1);
z=zeros(size(eigvector50{t},1)*size(eigvector50{t},2),1);
opp=1;
upp=1;
ttt=1;
for i=1:size(eigvector50{t},1)
    for j=1:size(eigvector50{t},2)
        x(opp)=i;
        y(upp)=j;
        z(ttt)=eigvector50{t}(i,j);
        opp=opp+1;
        upp=upp+1;
        ttt=ttt+1;
    end
end

%figure 
%plot3(x,y,z), title('Mainfold 50')

[a,b] = eigs(L0,100,'sa');
eigvalues100{t}=diag(real(b));
eigvector100{t}=real(a);


x=zeros(size(eigvector100{t},1)*size(eigvector100{t},2),1);
y=zeros(size(eigvector100{t},1)*size(eigvector100{t},2),1);
z=zeros(size(eigvector100{t},1)*size(eigvector100{t},2),1);
opp=1;
upp=1;
ttt=1;
for i=1:size(eigvector100{t},1)
    for j=1:size(eigvector100{t},2)
        x(opp)=i;
        y(upp)=j;
        z(ttt)=eigvector100{t}(i,j);
        opp=opp+1;
        upp=upp+1;
        ttt=ttt+1;
    end
end

%figure 
%plot3(x,y,z), title('Mainfold 100');

Kfet=zeros(180,1);
yu50=zeros(size(TestIm, 3),50);
yu100=zeros(size(TestIm, 3),100);
for i=1:size(TestIm, 3);
    sum=0;
    u=1;
    for z=1:size(XTrainNew, 3)
         Kol=exp(-((norm(TestIm(:,:,i)-XTrainNew(:,:,z)))^2)/2*sigma^2);
         if Kol~=1  
        
          if Kol>sigma
          Kfet(u)=1;
          sum=1+sum;
          else
          Kfet(u)=0;
          end
          u=u+1;
          
          end
     end
    yu50(i,:)=Kfet'*eigvector50{t}; 
    yu100(i,:)=Kfet'*eigvector100{t};
end

y50{t}=yu50;
y100{t}=yu100;

Mdl50{t} = fitcknn(eigvector50{t},Labels,'NumNeighbors',7);
Mdl100{t} = fitcknn(eigvector100{t},Labels,'NumNeighbors',7);

label_test_data50{t}= predict(Mdl50{t},y50{t});
label_test_data100{t}= predict(Mdl100{t},y100{t});
figure
subplot(1,2,1);
cm50{t} = confusionchart(Testlabels,label_test_data50{t});
title('M=50 expirement '+text(t));
subplot(1,2,2);
cm100{t} = confusionchart(Testlabels,label_test_data100{t}); 
title('M=100 expirement '+text(t));

end