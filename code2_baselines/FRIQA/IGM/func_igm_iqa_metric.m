function igm_val = func_igm_iqa_metric( img_ref, img_dst )

% transform color into gray
if size( img_ref, 3 ) == 3
    img_ref = double( rgb2gray( img_ref ) );
else
    img_ref = double( img_ref );
end
if size( img_dst, 3 ) == 3
    img_dst = double( rgb2gray( img_dst ) );
else
    img_dst = double( img_dst );
end
[ img_ref_dis, img_ref_prd] = func_igm_predict( img_ref );
[ img_dst_dis, img_dst_prd] = func_igm_predict( img_dst );
disorder_change = func_disorder_change( img_ref_dis, img_dst_dis );
edge_change = func_edge_height_change( img_ref_prd, img_dst_prd );
structure_change = func_structure_change( img_ref_prd, img_dst_prd );
alpha_val = func_alpha_value(img_ref, img_dst, img_ref_dis, img_dst_dis);
igm_val = disorder_change^alpha_val * (edge_change*structure_change)^(1-alpha_val);
end

function [ img_dis, img_prd] = func_igm_predict( img )

[row, col] = size( img );
sigma_value = 2*20.^2;
r = 3;
R = 10;
edge_threshold = 0.2;
mat_pad = padarray(img, [R+r,R+r], 'symmetric'); % pad the mat for edge processing
img_pad = mat_pad( R+1:end-R, R+1:end-R );
mat_edge = edge(mat_pad,'canny',edge_threshold);
edge_c = mat_edge( R+1:end-R, R+1:end-R );
ker = ones( 2*r+1 ) / ( 2*r+1 )^2;
img_reco = zeros( row, col );
weight_mat = zeros( row, col );
max_weight = ones( row, col ) * 0.01;
for u = -R : R
    for v = -R : R      
        if u==0 && v==0
            continue;
        end
        img_move = mat_pad( R+1+u:end-R+u, R+1+v:end-R+v );
        mat_dif = ( img_pad - img_move ).^2;
        sum_val = filter2( ker, mat_dif, 'valid' );        
        simi_val = exp( - sum_val ./ sigma_value );
        edge_m = mat_edge( R+1+u:end-R+u, R+1+v:end-R+v );
        sign_edge = ones( size( edge_c ) );
        sign_edge( edge_c == edge_m ) = 10;
        edge_simi =  sign_edge( r+1:end-r, r+1:end-r );       
        mat_simi = simi_val .* edge_simi;
        img_reco = img_reco + img_move(r+1:end-r, r+1:end-r).*mat_simi;
        weight_mat = weight_mat + mat_simi;
        max_weight( mat_simi > max_weight ) = weight_mat( mat_simi > max_weight );
    end
end
img_recon = img_reco + max_weight .* img;
weight_mat = weight_mat + max_weight;
img_reconst = uint8(img_recon ./ weight_mat);
img_dis = img - double( img_reconst );
%img_prd = double( img_reconst );
img_prd = img;
end

function disorder_change = func_disorder_change( img_ref, img_dst )
beta = [0.0448 0.2856 0.3001 0.2363 0.1333];
disorder_change = 1;
for level = 1 : length( beta )
    img_ref_ = imresize( img_ref, 0.5^(level-1) );
    img_dst_ = imresize( img_dst, 0.5^(level-1) );
    mse_val = mean( ( double( img_ref_(:) ) - double( img_dst_(:) ) ).^2 ) + 0.01;
    psnr_val = 10*log10( 255*255 / mse_val );
    disorder_change = disorder_change * psnr_val^beta( level );
end
disorder_change = ( disorder_change / (10*log10(255*255)) ).^(1/2);
end

function edge_change = func_edge_height_change( img1, img2 )
nlevs = 5;
K = 0.03;
% Use Analysis Low Pass filter from Biorthogonal 9/7 Wavelet
lod = [0.037828455507260; -0.023849465019560;  -0.110624404418440; ...
    0.377402855612830; 0.852698679008890;   0.377402855612830;  ...
    -0.110624404418440; -0.023849465019560; 0.037828455507260];
lpf = lod*lod';
lpf = lpf/sum(lpf(:));
img1 = double(img1);
img2 = double(img2);
edge_similar = zeros(nlevs,1);
% Scale 1 is the original image
edge_similar(1) = func_edge_change_index(img1,img2,K);
% Compute scales 2 through 5
for s=1:nlevs-1    
    % Low Pass Filter
    img1 = imfilter(img1,lpf,'symmetric','same');
    img2 = imfilter(img2,lpf,'symmetric','same');
    img1 = img1(1:2:end,1:2:end);
    img2 = img2(1:2:end,1:2:end);
    edge_similar(s+1) = func_edge_change_index(img1,img2,K);
end
beta = [0.0448 0.2856 0.3001 0.2363 0.1333]';
edge_change = prod(edge_similar.^beta);
end

function edg = func_edge_change_index(img1, img2, K)
L = 255;
C = (K*L)^2;
img1 = double(img1);
img2 = double(img2);
edge1 = func_edge( img1 );
edge2 = func_edge( img2 );
edg_map = ( 2*edge1.*edge2 + C ) ./ ( edge1.^2 + edge2.^2 + C );
edg = mean( edg_map(:) );
end

function edge_map = func_edge( img )
if ~isa( img, 'double' )
    img = double( img );
end
G1 = [0 0 0 0 0
   1 3 8 3 1
   0 0 0 0 0
   -1 -3 -8 -3 -1
   0 0 0 0 0];
G2=[0 0 1 0 0
   0 8 3 0 0
   1 3 0 -3 -1
   0 0 -3 -8 0
   0 0 -1 0 0];
G3=[0 0 1 0 0
   0 0 3 8  0
   -1 -3 0 3 1
   0 -8 -3 0 0
   0 0 -1 0 0];
G4=[0 1 0 -1 0
   0 3 0 -3 0
   0 8 0 -8 0
   0 3 0 -3 0
   0 1 0 -1 0];
% calculate the max grad
grad(:,:,1) = filter2(G1,img,'valid')/16;
grad(:,:,2) = filter2(G2,img,'valid')/16;
grad(:,:,3) = filter2(G3,img,'valid')/16;
grad(:,:,4) = filter2(G4,img,'valid')/16;
edge_map = max( abs(grad), [], 3 );
end

function structure_change = func_structure_change( img1, img2 )
nlevs = 5;
K = 0.03;
window = fspecial('gaussian',11,1.5);
% Use Analysis Low Pass filter from Biorthogonal 9/7 Wavelet
lod = [0.037828455507260; -0.023849465019560;  -0.110624404418440; ...
    0.377402855612830; 0.852698679008890;   0.377402855612830;  ...
    -0.110624404418440; -0.023849465019560; 0.037828455507260];
lpf = lod*lod';
lpf = lpf/sum(lpf(:));
img1 = double(img1);
img2 = double(img2);
content_similar = zeros(nlevs,1);
% Scale 1 is the original image
content_similar(1) = func_structure_change_index(img1,img2,K, window);
% Compute scales 2 through 5
for s=1:nlevs-1
    % Low Pass Filter
    img1 = imfilter(img1,lpf,'symmetric','same');
    img2 = imfilter(img2,lpf,'symmetric','same');   
    img1 = img1(1:2:end,1:2:end);
    img2 = img2(1:2:end,1:2:end);
    content_similar(s+1) = func_structure_change_index(img1,img2,K, window);
end
beta = [0.0448 0.2856 0.3001 0.2363 0.1333]';
structure_change = prod(content_similar.^beta);
end

function cont = func_structure_change_index(img1, img2, K, window)
L = 255;
C = (K*L)^2;
mu1   = filter2(window, img1, 'valid');
mu2   = filter2(window, img2, 'valid');
mu1_sq = mu1.*mu1;
mu2_sq = mu2.*mu2;
mu1_mu2 = mu1.*mu2;
sigma1_sq = filter2(window, img1.*img1, 'valid') - mu1_sq;
sigma2_sq = filter2(window, img2.*img2, 'valid') - mu2_sq;
sigma1 = real(sqrt(sigma1_sq));
sigma2 = real(sqrt(sigma2_sq));
sigma12 = filter2(window, img1.*img2, 'valid') - mu1_mu2;
cont_map = (sigma12 + C/2)./(sigma1.*sigma2+C/2);
cont = mean( cont_map(:) );
end

function alpha_val = func_alpha_value( img1_gray, img2_gray, img1_dis, img2_dis )
mse1 = mean( ( img1_gray(:) - img2_gray(:) ).^2 );
mse2 = mean( abs( ( img1_gray(:) - img2_gray(:) ).^2 - mse1 ) );
mseo = min( mse1, mse2 );
msed = mean( ( img1_dis(:) - img2_dis(:) ).^2 );
mse_11 = max( 0, mseo-30 )+1;
mse_22 = max( 0, msed-30 );
cc = 0.8;
coef = (mse_22+cc)./(mse_11+cc);
alpha_val = min( coef, 1 );
end

% end of this file