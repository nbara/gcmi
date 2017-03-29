function cx = copnorm_nan(x)
% COPNORM_NAN Copula normalisation with NaN propogation
%   cx = copnorm(x) returns standard normal samples with the same empirical
%   CDF value as the input. Operates along the first axis.
%   Equivalent to cx = norminv(ctransform(x))
%

x2d = x(:,:);
finidx = isfinite(x2d);
nanidx = ~finidx;
c2d = NaN(size(x2d));

if all(finidx(:))
  % no nans
  cx = copnorm(x);
  return
else
  anynans = any(nanidx,1);
  allnans = all(nanidx,1);
  % where there aren't any NaNs can copnorm directly
  finvars = ~anynans;
  c2d(:,finvars) = copnorm(x2d(:,finvars));
  if all( anynans == allnans )
    % no partial NaN variables so we are done
    cx = reshape(c2d, size(x));
    return
  else
    % some vars have partial nans, loop over variables
    partnans = find(anynans & ~allnans);
    for pi=1:length(partnans)
      idx = isfinite(x2d(:,partnans(pi)));
      c2d(idx, partnans(pi)) = copnorm(x2d(idx, partnans(pi)));
    end
  end
  cx = reshape(c2d, size(x));
end


