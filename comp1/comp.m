DATA = readtable('AAPL.csv', 'ReadVariableNames',0)


DATA_CLOSE=DATA(:,5)

length=size(DATA_CLOSE)
NUMBER_SAMPELS=floor(length(:,1)/3)

TRAIN=table2array(DATA_CLOSE(1:NUMBER_SAMPELS,:))
VAL=table2array(DATA_CLOSE(NUMBER_SAMPELS+1:NUMBER_SAMPELS*2,:))
TEST=table2array(DATA_CLOSE(NUMBER_SAMPELS*2+1:NUMBER_SAMPELS*3,:))


W(:,1)=[0; 0; 0]
X(1,:)=[TRAIN(3) TRAIN(2) TRAIN(1)]
Y(1)=TRAIN(4)
E(1)=Y(1)-X(1,:)*W(:,1)
mu=0.0001
tic
J(1)=E(1)^2

for i=1:NUMBER_SAMPELS-4
  X(i+1,:)=[TRAIN(i+2) TRAIN(i+1) TRAIN(i)]
  Y(i+1)=TRAIN(i+4)
  W(:,i+1)=W(:,i)+mu*X(i+1,:).'*E(i)
  E(i+1)=Y(i+1)-X(i+1,:)*W(:,i+1)
  J(i+1)=E(i+1)^2  
end

time=toc
figure
plot(J)

tic
WCONST=inv(X.'*X)*X.'*Y
time_2=toc

e=0
for i=1:NUMBER_SAMPELS-3
   X_VAL(i,:)=[VAL(i+2) VAL(i+1) VAL(i)]
   Y_VAL(i)=VAL(i+3)
   er(i)=Y_VAL(i)-X_VAL(i,:)*W(:,NUMBER_SAMPELS-3)
    
   e=e+er(i)^2 
end

MSE_VAL=e/(NUMBER_SAMPELS-3)
figure
plot(er)


e=0
for i=1:NUMBER_SAMPELS-3
   X_TEST(i,:)=[TEST(i+2) TEST(i+1) TEST(i)]
   Y_TEST(i)=TEST(i+3)
   er_test(i)=Y_TEST(i)-X_TEST(i,:)*W(:,NUMBER_SAMPELS-3)
    
   e=e+er_test(i)^2 
end

MSE_TEST=e/(NUMBER_SAMPELS-3)
figure
plot(er_test)



