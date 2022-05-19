function rec=reconstruct(X, eigenfaces , eigenfaces_value, mu)
r = 112;
c = 92;

summ=zeros(r*c,1)
for i = 1 : size(eigenfaces_value,2) 
   
   Xi=eigenfaces(:,i)'*double(X)/eigenfaces_value(i)
   summ(:,1)=Xi*eigenfaces(:,i)+summ(:,1)
end


rec=(summ+mu)
end