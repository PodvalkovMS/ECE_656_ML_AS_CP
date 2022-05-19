im=load('Lena.mat');
in=load('Boat.mat');
im=im.lena;
in=in.boat;
boat=in;
lena=im;

trash=4*ones(1,128);
Lena = mat2cell(lena, trash,trash);
boat = mat2cell(boat, trash,trash);
for i=1:128
    for j=1:128
        Lena{i,j}=reshape(Lena{i,j},[1,16]);
        boat{i,j}=reshape(boat{i,j},[1,16]);
    end
end
Lena = Lena';
Lena = reshape(Lena, [1,16384]);
Lena = vertcat(Lena{:});
Lena = Lena';

boat= boat';
boat = reshape(boat, [1,16384]);
boat = vertcat(boat{:});
boat = boat';



net=cell(5,1);
wiegth=cell(5,1);
wiegth_b=cell(5,8);
out=cell(5,1);
out_new=cell(5,1);
img=cell(5,8);
img_new=cell(5,8);
arc={[4 4],[4 8],[8 8],[8 16],[16 16]};
snr=zeros(5 ,8);
text=["[4 4]", "[4 8]" , "[8 8]" , "[8 16]" , "[16 16]"];
m=1;
for i=1:5
net{i} = selforgmap(arc{i}, 320);
net{i} = train(net{i},Lena);

wiegth{i}=net{i}.IW{1};
wiegth_b{i,1}=wiegth{i};
for l=2:8
wiegth_b{i,l}=uencode( ceil(wiegth{i}),l, max(ceil(wiegth{i}),[], 'all') );
end
out{i}=net{i}(Lena);
classes = vec2ind(out{i});


k = 0;

out_new{i}=net{i}(boat);


classes_new = vec2ind(out_new{i});

img_new{i} = zeros(512,512);

for l=1:8
    img{i, l} = zeros(512,512);
    img_new{i, l} = zeros(512,512);
    k=0;
for z = 1:4:512
    for j = 1: 4: 512
      k = k+1;
      winner = classes(1,k);
      winner_new = classes_new(1,k);
      img{i,l}(z:z+3,j:j+3) = reshape(wiegth_b{i,l}(winner,:),[4,4]);
      %img_new{i,l}(z:z+3,j:j+3) = reshape(wiegth_b{i,l}(winner_new,:),[4,4]);

    end
end
      snr(i,l) = 10*log10(norm(im)/norm(im-img{i,l}));
      fprintf('\n The SNR value is %0.4f  for %i net and %i bits', snr(i,l), i, l);
      
end
fprintf('\n');
figure(i)
subplot(2, 4, 1), imshow(img{i,1}, [0 255]), title('Original image')
for n=2:8
subplot(2, 4, n), imshow(img{i,n}, [0 (2^n)-1]), title(sprintf('Rec trained image with weights %d bpp', n))
end
sgtitle('Network '+text(i)) 

%figure(m+1)
%subplot(2, 4, 1), imshow(img_new{i,1}, [0 255]), title('Reconstract new image with weights original bits')
%for n=2:8
%subplot(2, 4, n), imshow(img_new{i,n}, [0 2^(n)-1]), title(sprintf('Reconstract new image with weights %d bits', n))
%end
%m=m+2;

end

x=2:1:8;
figure
plot(x, snr(1,2:8),x, snr(2,2:8),x, snr(3,2:8),x, snr(4,2:8),x, snr(5,2:8));
title('SNR vs BBP');
xlabel('BBP');
ylabel('SNR');
legend('net [4 4]', 'net [4 8]' , 'net [8 8]' , 'net [8 16]' , 'net [16 16]')





classes_new = vec2ind(out_new{1});

img_new_rec = zeros(512,512);

k=0
for z = 1:4:512
    for j = 1: 4: 512
      k = k+1;
      winner_new = classes_new(1,k);
     
      img_new_rec(z:z+3,j:j+3) = reshape(wiegth_b{1,7}(winner_new,:),[4,4]);

    end
end
figure
subplot(1, 2, 1), imshow(in, [0 255]), title('Original image')
subplot(1, 2, 2), imshow(img_new_rec, [0 127]), title('Rec new image with net [4 4] with 7 bbp')

