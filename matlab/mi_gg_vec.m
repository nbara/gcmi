function I = mi_gg_vec(x, y, biascorrect, demeaned, ignorenans)
% MI_GG Vectorized MI calculation between multiple Gaussian x variables and a 
%   common Gaussian y variable in bits
%   I = mi_gg_vec(x,y) returns the MI between two (possibly multidimensional)
%   Gassian variables, x and y, with bias correction.
%   size(x) = [Ntrl Nvec Ndim]
%   so each output I(i) = mi_gg_vec(squeeze(x(:,i,:)), y);
%
%   biascorrect : true / false option (default true) which specifies whether
%   bias correction should be applied to the esimtated MI.
%   demeaned : false / true option (default false) which specifies whether the
%   input data already has zero mean (true if it has been copula-normalized)

% ensure samples first axis for vectors
if isvector(x)
  x = x(:);
end
if isvector(y)
  y = y(:);
end
if ndims(x)>3 
  error('mi_gg_vec: x input array should be 3d')
end
if ndims(y)~=2
  error('mi_gg_vec: y input should be 2d')
end

% default option values
if nargin<3
    biascorrect = true;
end
if nargin<4
    demeaned = false;
end
if nargin<5
    ignorenans = false;
end

Ntrl = size(x,1);
Nvec = size(x,2);
Nvarx = size(x,3);
Nvary = size(y,2);
Nvarxy = Nvarx + Nvary;
Ntrly = size(y,1);
if ignorenans
  mysum = @nansum;
  % only include trials/variables where all dimensions are finite
  finidx = all(isfinite(x),3);
  % number of finite trials for each variable
  Ntrl = squeeze(sum(finidx,1))';
  % set all dimensions to nan when one is
  x(~finidx(:,:,ones(1,Nvarx))) = NaN;
else
  mysum = @sum;
  Ntrl = repmat(size(x,1),[size(x,2) 1]);
end

if size(y,1) ~= size(x,1)
    error('mi_gg_vec: number of trials do not match')
end

% demean data if required
if ~demeaned
    x = bsxfun(@minus,x,bsxfun(@rdivide,mysum(x,1),Ntrl'));
    y = bsxfun(@minus,y,sum(y)./Ntrly);
end

Cx = zeros(Nvec, Nvarx, Nvarx);
Cxy = zeros(Nvec, Nvarxy, Nvarxy);

Cy = y'*y / (Ntrly - 1);

% Cx and Cx part of Cxy
for vi1=1:Nvarx
  x1 = x(:,:,vi1);
  thsV = mysum(x1.^2);
  Cx(:,vi1,vi1) = thsV;
  Cxy(:,vi1,vi1) = thsV;
  
  for vi2=(vi1+1):Nvarx
    x2 = x(:,:,vi2);
    thsC = mysum(x1.*x2);
    Cx(:,vi1,vi2) = thsC;
    Cx(:,vi2,vi1) = thsC;
    Cxy(:,vi1,vi2) = thsC;
    Cxy(:,vi2,vi1) = thsC;
  end
end

% Cx = Cx / (Ntrl-1);
Cx = bsxfun(@rdivide, Cx, Ntrl-1);

% Cxy part of Cxy
for vi1=1:Nvarx
  x1 = x(:,:,vi1);
  for vi2=1:Nvary
    y1 = y(:,vi2);
    thsC = mysum(bsxfun(@times,x1,y1));
    Cxy(:,Nvarx+vi2,vi1) = thsC;
    Cxy(:,vi1,Nvarx+vi2) = thsC;
  end
end

% Cxy = Cxy ./ (Ntrl-1);
Cxy = bsxfun(@rdivide, Cxy, Ntrl-1);

% Cy part of Cxy
% NB if ignorenans this might be calculated with different numbers of
% trials the Cxx, Cxy blocks
for vi1=1:Nvary
  Cxy(:,Nvarx+vi1,Nvarx+vi1) = Cy(vi1,vi1);
  for vi2=(vi1+1):Nvary
    Cxy(:,Nvarx+vi1,Nvarx+vi2) = Cy(vi1,vi2);
    Cxy(:,Nvarx+vi2,Nvarx+vi1) = Cy(vi1,vi2);
  end
end

% entropies in nats
% normalisations cancel for information
chCy = chol(Cy);
HY = sum(log(diag(chCy))); % + 0.5*Nvary*log(2*pi*exp(1));

chCx = vecchol(Cx);
chCxy = vecchol(Cxy);
HX = zeros(Nvec,1);
HXY = zeros(Nvec,1);
for vi=1:Nvarx
%   HX = HX + shiftdim(log(Cx(:,vi,vi)));
  HX = HX + log(chCx(:,vi,vi));
end
for vi=1:Nvarxy
%   HXY = HXY + shiftdim(log(Cxy(:,vi,vi)));
  HXY = HXY + log(chCxy(:,vi,vi));
end

ln2 = log(2);
if biascorrect
    psiterms_x = psi((Ntrl - (1:Nvarxy))/2) / 2;
    dterm_x = (ln2 - log(Ntrl-1)) / 2;
%     HX = (HX - Nvarx*dterm - sum(psiterms(1:Nvarx)));
%     HY = (HY - Nvary*dterm - sum(psiterms(1:Nvary)));
%     HXY = (HXY - Nvarxy*dterm - sum(psiterms));
    HXbias = Nvarx*dterm_x + sum(psiterms_x(:,1:Nvarx),2);
    HXYbias = Nvarxy*dterm_x + sum(psiterms_x,2);

    psiterms_y = psi((Ntrly - (1:Nvary))/2) / 2;
    dterm_y = (ln2 - log(Ntrly-1)) / 2;
    HYbias = Nvary*dterm_y + sum(psiterms_y);
    Ibias = HXbias + HYbias - HXYbias;
else
    Ibias = 0;
end

% convert to bits
% I = (HX + HY - HXY) / ln2;
I = (bsxfun(@plus,HX-HXY, HY) - Ibias) ./ ln2;

